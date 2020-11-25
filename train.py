import argparse
import math
import os
import time

import torch
import torch.distributed as dist
from numpy import finfo
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from distributed import apply_gradient_allreduce
from hparams import create_hparams
from models import load_model
from utils.data_reader import TextMelCollate, TextMelLoader, DynamicBatchSampler
from utils.data_reader_refine import TextMelLoader_refine
from utils.logger import Tacotron2Logger
from utils.utils import get_checkpoint_path, learning_rate_decay, print_rank
from utils import ValueWindow


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams) if not hparams.is_partial_refine else TextMelLoader_refine(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams) if hparams.validation_files != '' else None
    collate_fn = TextMelCollate(hparams)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    if hparams.batch_criterion == 'frame':
        batch_sampler = DynamicBatchSampler(train_sampler, frames_threshold=hparams.batch_size)
        train_loader = DataLoader(trainset,
                                batch_sampler=batch_sampler,
                                num_workers=hparams.numberworkers,
                                pin_memory=True,
                                collate_fn=collate_fn)
    elif hparams.batch_criterion == 'utterance':
        train_loader = DataLoader(trainset,
                                sampler=train_sampler, batch_size=hparams.batch_size,
                                num_workers=hparams.numberworkers, shuffle=shuffle,
                                pin_memory=True,
                                drop_last=False,
                                collate_fn=collate_fn)
    else:
        raise ValueError("batch criterion not supported: %s." % hparams.batch_criterion)

    return train_loader, valset, collate_fn, trainset


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(log_directory)
    else:
        logger = None
    return logger


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer, hparams, style_list=None):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    
    if hparams.is_partial_refine and style_list:
        model_dict = checkpoint_dict['state_dict']
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in style_list}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(checkpoint_dict['state_dict'], strict=False)
        try:
            optimizer.load_state_dict(checkpoint_dict['optimizer'])
        except:
            print("Can't use old optimizer")

    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, valset, iteration, batch_criterion, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset,
                                sampler=val_sampler, batch_size=1,
                                num_workers=0, shuffle=False,
                                pin_memory=False,
                                collate_fn=collate_fn)
        '''if batch_criterion == 'frame':
            batch_sampler = DynamicBatchSampler(val_sampler, frames_threshold=batch_size)
            val_loader = DataLoader(valset,
                                    batch_sampler=batch_sampler,
                                    num_workers=0,
                                    pin_memory=False,
                                    collate_fn=collate_fn)
        elif batch_criterion == 'utterance':
            val_loader = DataLoader(valset,
                                    sampler=val_sampler, batch_size=batch_size,
                                    num_workers=0, shuffle=False,
                                    pin_memory=False,
                                    collate_fn=collate_fn)
        else:
            raise ValueError("batch criterion not supported: %s." % batch_criterion)'''

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            _, loss = model(batch, iteration)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    if rank == 0:
        print("{} Validation loss: {:9f}  ".format(iteration, val_loss))
        model.log_validation(logger, val_loss, iteration)
    model.train()


def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams, refine_from):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.initial_learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    if hparams.use_GAN and hparams.GAN_type=='lsgan':
        from discriminator import Lsgan_Loss, Calculate_Discrim
        model_D = Calculate_Discrim(hparams).cuda() if torch.cuda.is_available() else Calculate_Discrim(hparams)
        lsgan_loss = Lsgan_Loss(hparams)
        optimizer_D = torch.optim.Adam(model_D.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)
    
    if hparams.use_GAN and hparams.GAN_type=='wgan-gp':
        from discriminator import Wgan_GP, GP
        model_D = Wgan_GP(hparams).cuda() if torch.cuda.is_available() else Wgan_GP(hparams)
        calc_gradient_penalty = GP(hparams).cuda() if torch.cuda.is_available() else GP(hparams)
        optimizer_D = torch.optim.Adam(model_D.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)   

    if hparams.is_partial_refine:
        refine_list=['speaker_embedding.weight',
        'spkemb_projection.weight',
        'spkemb_projection.bias',
        'projection.weight',
        'projection.bias',
        'encoder.encoders.0.norm1.w.weight',
        'encoder.encoders.0.norm1.w.bias',
        'encoder.encoders.0.norm1.b.weight',
        'encoder.encoders.0.norm1.b.bias',
        'encoder.encoders.0.norm2.w.weight',
        'encoder.encoders.0.norm2.w.bias',
        'encoder.encoders.0.norm2.b.weight',
        'encoder.encoders.0.norm2.b.bias',
        'encoder.encoders.1.norm1.w.weight',
        'encoder.encoders.1.norm1.w.bias',
        'encoder.encoders.1.norm1.b.weight',
        'encoder.encoders.1.norm1.b.bias',
        'encoder.encoders.1.norm2.w.weight',
        'encoder.encoders.1.norm2.w.bias',
        'encoder.encoders.1.norm2.b.weight',
        'encoder.encoders.1.norm2.b.bias',
        'encoder.encoders.2.norm1.w.weight',
        'encoder.encoders.2.norm1.w.bias',
        'encoder.encoders.2.norm1.b.weight',
        'encoder.encoders.2.norm1.b.bias',
        'encoder.encoders.2.norm2.w.weight',
        'encoder.encoders.2.norm2.w.bias',
        'encoder.encoders.2.norm2.b.weight',
        'encoder.encoders.2.norm2.b.bias',
        'encoder.encoders.3.norm1.w.weight',
        'encoder.encoders.3.norm1.w.bias',
        'encoder.encoders.3.norm1.b.weight',
        'encoder.encoders.3.norm1.b.bias',
        'encoder.encoders.3.norm2.w.weight',
        'encoder.encoders.3.norm2.w.bias',
        'encoder.encoders.3.norm2.b.weight',
        'encoder.encoders.3.norm2.b.bias',
        'encoder.encoders.4.norm1.w.weight',
        'encoder.encoders.4.norm1.w.bias',
        'encoder.encoders.4.norm1.b.weight',
        'encoder.encoders.4.norm1.b.bias',
        'encoder.encoders.4.norm2.w.weight',
        'encoder.encoders.4.norm2.w.bias',
        'encoder.encoders.4.norm2.b.weight',
        'encoder.encoders.4.norm2.b.bias',
        'encoder.encoders.5.norm1.w.weight',
        'encoder.encoders.5.norm1.w.bias',
        'encoder.encoders.5.norm1.b.weight',
        'encoder.encoders.5.norm1.b.bias',
        'encoder.encoders.5.norm2.w.weight',
        'encoder.encoders.5.norm2.w.bias',
        'encoder.encoders.5.norm2.b.weight',
        'encoder.encoders.5.norm2.b.bias',
        'encoder.after_norm.w.weight',
        'encoder.after_norm.w.bias',
        'encoder.after_norm.b.weight',
        'encoder.after_norm.b.bias',
        'duration_predictor.norm.0.w.weight',
        'duration_predictor.norm.0.w.bias',
        'duration_predictor.norm.0.b.weight',
        'duration_predictor.norm.0.b.bias',
        'duration_predictor.norm.1.w.weight',
        'duration_predictor.norm.1.w.bias',
        'duration_predictor.norm.1.b.weight',
        'duration_predictor.norm.1.b.bias',
        'decoder.encoders.0.norm1.w.weight',
        'decoder.encoders.0.norm1.w.bias',
        'decoder.encoders.0.norm1.b.weight',
        'decoder.encoders.0.norm1.b.bias',
        'decoder.encoders.0.norm2.w.weight',
        'decoder.encoders.0.norm2.w.bias',
        'decoder.encoders.0.norm2.b.weight',
        'decoder.encoders.0.norm2.b.bias',
        'decoder.encoders.1.norm1.w.weight',
        'decoder.encoders.1.norm1.w.bias',
        'decoder.encoders.1.norm1.b.weight',
        'decoder.encoders.1.norm1.b.bias',
        'decoder.encoders.1.norm2.w.weight',
        'decoder.encoders.1.norm2.w.bias',
        'decoder.encoders.1.norm2.b.weight',
        'decoder.encoders.1.norm2.b.bias',
        'decoder.encoders.2.norm1.w.weight',
        'decoder.encoders.2.norm1.w.bias',
        'decoder.encoders.2.norm1.b.weight',
        'decoder.encoders.2.norm1.b.bias',
        'decoder.encoders.2.norm2.w.weight',
        'decoder.encoders.2.norm2.w.bias',
        'decoder.encoders.2.norm2.b.weight',
        'decoder.encoders.2.norm2.b.bias',
        'decoder.encoders.3.norm1.w.weight',
        'decoder.encoders.3.norm1.w.bias',
        'decoder.encoders.3.norm1.b.weight',
        'decoder.encoders.3.norm1.b.bias',
        'decoder.encoders.3.norm2.w.weight',
        'decoder.encoders.3.norm2.w.bias',
        'decoder.encoders.3.norm2.b.weight',
        'decoder.encoders.3.norm2.b.bias',
        'decoder.encoders.4.norm1.w.weight',
        'decoder.encoders.4.norm1.w.bias',
        'decoder.encoders.4.norm1.b.weight',
        'decoder.encoders.4.norm1.b.bias',
        'decoder.encoders.4.norm2.w.weight',
        'decoder.encoders.4.norm2.w.bias',
        'decoder.encoders.4.norm2.b.weight',
        'decoder.encoders.4.norm2.b.bias',
        'decoder.encoders.5.norm1.w.weight',
        'decoder.encoders.5.norm1.w.bias',
        'decoder.encoders.5.norm1.b.weight',
        'decoder.encoders.5.norm1.b.bias',
        'decoder.encoders.5.norm2.w.weight',
        'decoder.encoders.5.norm2.w.bias',
        'decoder.encoders.5.norm2.b.weight',
        'decoder.encoders.5.norm2.b.bias',
        'decoder.after_norm.w.weight',
        'decoder.after_norm.w.bias',
        'decoder.after_norm.b.weight',
        'decoder.after_norm.b.bias']
        if hparams.is_refine_style:
            style_list= ['gst.ref_enc.convs.0.weight',
        'gst.ref_enc.convs.1.weight',
        'gst.ref_enc.convs.1.bias',
        'gst.ref_enc.convs.3.weight',
        'gst.ref_enc.convs.4.weight',
        'gst.ref_enc.convs.4.bias',
        'gst.ref_enc.convs.6.weight',
        'gst.ref_enc.convs.7.weight',
        'gst.ref_enc.convs.7.bias',
        'gst.ref_enc.convs.9.weight',
        'gst.ref_enc.convs.10.weight',
        'gst.ref_enc.convs.10.bias',
        'gst.ref_enc.convs.12.weight',
        'gst.ref_enc.convs.13.weight',
        'gst.ref_enc.convs.13.bias',
        'gst.ref_enc.convs.15.weight',
        'gst.ref_enc.convs.16.weight',
        'gst.ref_enc.convs.16.bias',
        'gst.ref_enc.gru.weight_ih_l0,'
        'gst.ref_enc.gru.weight_hh_l0',
        'gst.ref_enc.gru.bias_ih_l0',
        'gst.ref_enc.gru.bias_hh_l0',
        'gst.stl.gst_embs',
        'gst.stl.mha.linear_q.weight',
        'gst.stl.mha.linear_q.bias',
        'gst.stl.mha.linear_k.weight',
        'gst.stl.mha.linear_k.bias',
        'gst.stl.mha.linear_v.weight',
        'gst.stl.mha.linear_v.bias',
        'gst.stl.mha.linear_out.weight',
        'gst.stl.mha.linear_out.bias',
        'gst.choosestl.choose_mha.linear_q.weight',
        'gst.choosestl.choose_mha.linear_q.bias',
        'gst.choosestl.choose_mha.linear_k.weight',
        'gst.choosestl.choose_mha.linear_k.bias',
        'gst.choosestl.choose_mha.linear_v.weight',
        'gst.choosestl.choose_mha.linear_v.bias',
        'gst.choosestl.choose_mha.linear_out.weight',
        'gst.choosestl.choose_mha.linear_out.bias',
        'gst_projection.weight',
        'gst_projection.bias'
        ]
        refine_list += style_list
    
    for name, param in model.named_parameters():
        if hparams.is_partial_refine:
            if name in refine_list:
                param.requires_grad = True 
            else:
                param.requires_grad = False
        print(name, param.requires_grad, param.shape)

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)
        if hparams.use_GAN:
            model_D = apply_gradient_allreduce(model_D)

    logger = prepare_directories_and_logger(output_directory, log_directory, rank)

    train_loader, valset, collate_fn, trainset = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if not checkpoint_path:
        checkpoint_path = get_checkpoint_path(output_directory) if not hparams.is_partial_refine else get_checkpoint_path(refine_from)
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer, hparams, style_list=style_list if hparams.is_refine_style else None)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration = (iteration + 1)  if not hparams.is_partial_refine else 0# next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader))) if not hparams.is_partial_refine else 0

    model.train()
    if hparams.use_GAN:
        model_D.train()
    else:
        hparams.use_GAN = True
        hparams.Generator_pretrain_step = hparams.iters
    is_overflow = False
    epoch = epoch_offset
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    # ================ MAIN TRAINNIG LOOP! ===================
    while iteration <= hparams.iters:
        # print("Epoch: {}".format(epoch))
        if hparams.distributed_run and hparams.batch_criterion == 'utterance':
            train_loader.sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            learning_rate = learning_rate_decay(iteration, hparams)
            if hparams.use_GAN:

                # Discriminator turn
                if iteration > hparams.Generator_pretrain_step:
                    for param_group in optimizer_D.param_groups:
                        param_group['lr'] = learning_rate
                    optimizer.zero_grad()
                    optimizer_D.zero_grad()
                    for name, param in model.named_parameters():
                        param.requires_grad = False
                    for name, param in model_D.named_parameters():
                        param.requires_grad = True 

                    loss, loss_dict, weight, pred_outs, ys, olens = model(*model._parse_batch(batch,hparams,utt_mels=trainset.utt_mels if hparams.is_refine_style else None))
                    if hparams.GAN_type=='lsgan':
                        discrim_gen_output, discrim_target_output = model_D(pred_outs + (torch.randn(pred_outs.size()).cuda() if hparams.add_noise else 0), ys + (torch.randn(pred_outs.size()).cuda() if hparams.add_noise else 0), olens)
                        loss_D = lsgan_loss(discrim_gen_output, discrim_target_output, train_object='D')
                        loss_G = lsgan_loss(discrim_gen_output, discrim_target_output, train_object='G')                
                        loss_D.backward(retain_graph=True)
                    if hparams.GAN_type=='wgan-gp':
                        D_real = model_D(ys, olens)
                        D_real = -D_real.mean()
                        D_real.backward(retain_graph=True)
                        D_fake = model_D(pred_outs, olens)
                        D_fake = D_fake.mean()
                        D_fake.backward()
                        gradient_penalty = calc_gradient_penalty(model_D, ys.data, pred_outs.data, olens.data)
                        gradient_penalty.backward()
                        D_cost = D_real + D_fake + gradient_penalty
                        Wasserstein_D = -D_real - D_fake
                    grad_norm_D = torch.nn.utils.clip_grad_norm_(model_D.parameters(), hparams.grad_clip_thresh)
                    optimizer_D.step()
                    print('\n')
                    if hparams.GAN_type=='lsgan':
                        print("Epoch:{} step:{} loss_D: {:>9.6f}, loss_G: {:>9.6f}, Grad Norm: {:>9.6f}".format(epoch, iteration, loss_D, loss_G, grad_norm_D))
                    if hparams.GAN_type=='wgan-gp':
                        print("Epoch:{} step:{} D_cost: {:>9.6f}, Wasserstein_D: {:>9.6f}, GP: {:>9.6f}, Grad Norm: {:>9.6f}".format(epoch, iteration, D_cost, Wasserstein_D, gradient_penalty, grad_norm_D))

                # Generator turn
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate 
                optimizer.zero_grad()
                if iteration > hparams.Generator_pretrain_step:
                    for name, param in model.named_parameters():
                        if hparams.is_partial_refine:
                            if name in refine_list:
                                param.requires_grad = True
                        else:
                            param.requires_grad = True                                 
                    for name, param in model_D.named_parameters():
                        param.requires_grad = False
                    optimizer_D.zero_grad()
                loss, loss_dict, weight, pred_outs, ys, olens = model(*model._parse_batch(batch,hparams,utt_mels=trainset.utt_mels if hparams.is_refine_style else None))
                if iteration > hparams.Generator_pretrain_step:
                    if hparams.GAN_type=='lsgan':
                        discrim_gen_output, discrim_target_output = model_D(pred_outs, ys, olens)
                        loss_D = lsgan_loss(discrim_gen_output, discrim_target_output, train_object='D')
                        loss_G = lsgan_loss(discrim_gen_output, discrim_target_output, train_object='G')
                    if hparams.GAN_type=='wgan-gp':
                        loss_G = model_D(pred_outs, olens)
                        loss_G = -loss_G.mean()
                    loss = loss + loss_G*hparams.GAN_alpha*abs(loss.item()/loss_G.item())
                if hparams.distributed_run:
                    reduced_loss = reduce_tensor(loss.data, n_gpus).item()
                    if loss_dict:
                        for key in loss_dict:
                            loss_dict[key] = reduce_tensor(loss_dict[key].data, n_gpus).item()
                else:
                    reduced_loss = loss.item()
                    if loss_dict:
                        for key in loss_dict:
                            loss_dict[key] = loss_dict[key].item()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)
                optimizer.step()
                duration = time.perf_counter() - start
                time_window.append(duration)
                loss_window.append(reduced_loss)
                if not is_overflow and (rank == 0):
                    if iteration % hparams.log_per_checkpoint == 0:
                        if hparams.GAN_type=='lsgan':
                            print("Epoch:{} step:{} Train loss: {:>9.6f}, avg loss: {:>9.6f}, Grad Norm: {:>9.6f}, {:>5.2f}s/it, {:s} loss: {:>9.6f}, D_loss: {:>9.6f}, G_loss: {:>9.6f}, duration loss: {:>9.6f}, ssim loss: {:>9.6f}, lr: {:>4}".format(
                            epoch, iteration, reduced_loss, loss_window.average, grad_norm, time_window.average, hparams.loss_type, loss_dict[hparams.loss_type], loss_D.item() if iteration > hparams.Generator_pretrain_step else 0, loss_G.item() if iteration > hparams.Generator_pretrain_step else 0, loss_dict["duration_loss"], loss_dict["ssim_loss"], learning_rate))
                        if hparams.GAN_type=='wgan-gp':
                            print("Epoch:{} step:{} Train loss: {:>9.6f}, avg loss: {:>9.6f}, Grad Norm: {:>9.6f}, {:>5.2f}s/it, {:s} loss: {:>9.6f}, G_loss: {:>9.6f}, duration loss: {:>9.6f}, ssim loss: {:>9.6f}, lr: {:>4}".format(
                            epoch, iteration, reduced_loss, loss_window.average, grad_norm, time_window.average, hparams.loss_type, loss_dict[hparams.loss_type], loss_G.item() if iteration > hparams.Generator_pretrain_step else 0, loss_dict["duration_loss"], loss_dict["ssim_loss"], learning_rate))
                        logger.log_training(
                            reduced_loss, grad_norm, learning_rate, duration, iteration, loss_dict)

            if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
                if valset is not None:
                    validate(model, valset, iteration, hparams.batch_criterion,
                            hparams.batch_size, n_gpus, collate_fn, logger,
                            hparams.distributed_run, rank)
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}_refine_{}".format(iteration, hparams.training_files.split('/')[-2].split('_')[-1]) if hparams.is_partial_refine else "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)

            iteration += 1
            torch.cuda.empty_cache()
        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-model-path', dest='output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log-dir', dest='log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--refine_from', dest='refine_from', type=str, default=None, required=False,
                        help='load model to be refined')
    parser.add_argument('--warm_start', action='store_true', 
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    parser.add_argument('--hparams_json', type=str,
                        required=False, help='hparams json file')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams, args.hparams_json)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
    torch.backends.cudnn.deterministic = True

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)


    train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams, args.refine_from)

#python train.py -o=./checkpoint -l=./logdir --refine_from=./checkpoint/ --hparams='use_gst=True,is_refine_style=True,distributed_run=False,fp16_run=False,cudnn_benchmark=False,iters=40000,batch_criterion=utterance,batch_size=4,is_multi_styles=False,is_multi_speakers=True,is_spk_layer_norm=True,use_ssim_loss=True,loss_type=L1,is_partial_refine=True,use_GAN=True,GAN_alpha=0.1'