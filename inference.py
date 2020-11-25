import argparse
import os, io
import shutil
import sys, json
import zipfile
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import soundfile
import torch
from scipy.io.wavfile import read as wavread
from hparams import create_hparams
from models import load_model
from utils.utils import _plot_and_save, save_htk_data
from utils.audio import Audio
from utils.data_reader import _read_meta_yyh
from utils.data_reader_refine import TextMelLoader_refine

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_model(model_path, hparams):
    # Load model from checkpoint
    model = load_model(hparams)
    model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
    for name, param in model.named_parameters():
        print(str(name), param.requires_grad, param.shape) 
    model.to(device).eval()
    return model

class Synthesizer():
    def __init__(self, model_path, out_dir, text_file, sil_file, use_griffin_lim, hparams):
        self.model_path = model_path
        self.out_dir = out_dir
        self.text_file = text_file
        self.sil_file = sil_file
        self.use_griffin_lim = use_griffin_lim
        self.hparams = hparams

        self.model = get_model(model_path, hparams)
        self.audio_class = Audio(hparams)

        if hparams.use_phone:
            from text.phones import Phones
            phone_class = Phones(hparams.phone_set_file)
            self.text_to_sequence = phone_class.text_to_sequence
        else:
            from text import text_to_sequence
            self.text_to_sequence = text_to_sequence

        # self.out_png_dir = os.path.join(self.out_dir, 'png')
        # os.makedirs(self.out_png_dir, exist_ok=True)

        self.out_wav_dir = os.path.join(self.out_dir, 'wav')
        os.makedirs(self.out_wav_dir, exist_ok=True)

    def get_inputs(self, meta_data):
        hparams = self.hparams
        SEQUENCE_ = []
        SPEAKERID_ = []
        STYLE_ID_ = []
        FILENAME_ = []
        # Prepare text input
        for i in range(len(meta_data)):
            filename = meta_data[i].strip().split('|')[1]
            print('Filename=', filename)

            phone_text = meta_data[i].strip().split('|')[-1]
            print('Text=', phone_text)

            speaker_id = int(meta_data[i].strip().split('|')[-2])
            print('SpeakerID=', speaker_id)
            
            sequence = np.array(self.text_to_sequence(meta_data[i].strip().split('|')[-1], ['english_cleaners']))   # [None, :]
            print(sequence)
            
            sequence = torch.autograd.Variable(torch.from_numpy(sequence)).to(device).long()
            
            speaker_id = torch.LongTensor([speaker_id]).to(device) if hparams.is_multi_speakers else None

            style_id = torch.LongTensor([int(meta_data[i].strip().split('|')[1])]).to(device) if hparams.is_multi_styles else None

            SEQUENCE_.append(sequence)
            SPEAKERID_.append(speaker_id)
            STYLE_ID_.append(style_id)
            FILENAME_.append(filename)

        return SEQUENCE_, SPEAKERID_, STYLE_ID_, FILENAME_

    def gen_mel(self, meta_data):
        SEQUENCE_, SPEAKERID_, STYLE_ID_, FILENAME_ = self.get_inputs(meta_data)
        MEL_OUTOUTS_ = []
        FILENAME_NEW_ = []
        # Decode text input and plot results
        with torch.no_grad():
            for i in range(len(SEQUENCE_)):
                mel_outputs, _, _ = self.model.inference(text=SEQUENCE_[i], spk_ids=SPEAKERID_[i], utt_mels=TextMelLoader_refine(hparams.training_files, hparams).utt_mels)
                MEL_OUTOUTS_.append(mel_outputs.transpose(0, 1).float().data.cpu().numpy())  # (dim, length)
                FILENAME_NEW_.append('spkid_' + str(SPEAKERID_[i].item()) + '_filenum_' + FILENAME_[i])
                print('mel_outputs.shape=',mel_outputs.shape)
        # image_path = os.path.join(self.out_png_dir, "{}_spec_stop.png".format(filename))
        # fig, axes = plt.subplots(1, 1, figsize=(8,8))
        # axes.imshow(mel_outputs, aspect='auto', origin='bottom', interpolation='none')
        # plt.savefig(image_path, format='png')
        # plt.close()
        return MEL_OUTOUTS_, FILENAME_NEW_

    def gen_wav_griffin_lim(self, mel_outputs, filename):
        grf_wav = self.audio_class.inv_mel_spectrogram(mel_outputs)
        grf_wav = self.audio_class.inv_preemphasize(grf_wav)
        wav_path = os.path.join(self.out_wav_dir, "{}-gl.wav".format(filename))
        self.audio_class.save_wav(grf_wav, wav_path)

    def inference_f(self):
        # print(meta_data['n'])
        meta_data = _read_meta_yyh(self.text_file)
        MEL_OUTOUTS_, FILENAME_NEW_ = self.gen_mel(meta_data)
        for i in range(len(MEL_OUTOUTS_)):
            np.save(os.path.join(self.out_wav_dir, "{}.npy".format(FILENAME_NEW_[i] + '_' + self.model_path.split('/')[-1])), MEL_OUTOUTS_[i].transpose(1, 0))
            if self.use_griffin_lim:
                self.gen_wav_griffin_lim(MEL_OUTOUTS_[i], FILENAME_NEW_[i] + '_' + self.model_path.split('/')[-1])
        
        source_dir = self.out_wav_dir
        target_dir_npy = r'../../../Melgan_pipeline/melgan_file/gen_npy'
        target_dir_wav = r'../../../Melgan_pipeline/melgan_file/gen_wav'
        target_dir_wav_16k = r'../../../Melgan_pipeline/melgan_file/gen_wav_16k'
        
        for file in sorted(os.listdir(source_dir)):
            temp_file_dir = source_dir + os.sep + file
            shutil.copy(temp_file_dir,target_dir_npy)
            os.remove(temp_file_dir)
        
        os.system('cd ../../../Melgan_pipeline/melgan_file/scripts/ && python generate_from_folder_mels_4-4_npy.py --load_path=../../personalvoice/melgan16k --folder=../gen_npy --save_path=../gen_wav')
        os.system('cd ../../../Tools/16kConverter/ && python Convert16k.py --wave_path=../../Melgan_pipeline/melgan_file/gen_wav --output_dir=../../Melgan_pipeline/melgan_file/gen_wav_16k')
        
        for file in sorted(os.listdir(target_dir_npy)):
            temp_file_dir = target_dir_npy + os.sep + file
            os.remove(temp_file_dir)
        
        for file in sorted(os.listdir(target_dir_wav)):
            temp_file_dir = target_dir_wav + os.sep + file
            os.remove(temp_file_dir)

        for file in sorted(os.listdir(target_dir_wav_16k)):
            temp_file_dir = target_dir_wav_16k + os.sep + file
            shutil.copy(temp_file_dir,source_dir)
            os.remove(temp_file_dir)

        return print('finished')

if __name__ == '__main__':
    def str2bool(s):
        s = s.lower()
        assert s in ["true", "false"]
        return {'t': True, 'f': False}[s[0]]

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--model_path', type=str, default="checkpoint_refine/checkpoint_3000_refine_SDIM02_20",
                        help='transformer checkpoint path')
    parser.add_argument('-t', '--text_file', type=str, default="test_txt_others.txt",
                        help='text file')
    parser.add_argument('-o', '--out_dir', type=str, default="gen_wav",
                        required=False, help='output dir')
    parser.add_argument('-s', '--sil_file', type=str, default="utils/Seed_16k.wav",
                        required=False, help='silence audio')
    parser.add_argument('--use_griffin_lim', type=str2bool, default=False,
                        help='whether generate wav using grifflin lim')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    parser.add_argument('--hparams_json', type=str,
                        required=False, help='hparams json file')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams, args.hparams_json)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    os.makedirs(args.out_dir, exist_ok=True)
    synthesizer = Synthesizer(args.model_path, args.out_dir, args.text_file, args.sil_file,
                              args.use_griffin_lim, hparams)
    synthesizer.inference_f()

# python inference.py --hparams='use_gst=True,is_refine_style=True,training_files=./refine_data_Jeanne/training_with_mel_frame_refine.txt,mel_dir=./refine_data_Jeanne/'