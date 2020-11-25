import argparse
import os, io
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
from utils.data_reader import _read_meta, _read_meta_yyh
from duration_calculator import DurationCalculator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_duration_matrix(char_text_dir, duration_tensor, save_mode='phone'):
    fid = open(char_text_dir)
    text = fid.readlines()
    fid.close()
    text_utt = list(text[0].strip().split('|')[-1])
    frame_length = torch.sum(duration_tensor)
    duration_tensor_matrix = torch.zeros(frame_length, 80)
    idx = 0
    
    if save_mode == 'char':
        for i in range(len(duration_tensor)-1):
            idx += duration_tensor[i]
            duration_tensor_matrix[idx-1,:]=1

    
    if save_mode == 'phone':
        for i in range(len(text_utt)-1):
            idx += duration_tensor[i]
            if text_utt[i+1] == ' ':
                duration_tensor_matrix[idx-1,:]=1
                duration_tensor_matrix[idx+duration_tensor[i+1]-1,:]=1

    return duration_tensor_matrix



def get_model(model_path, hparams):
    # Load model from checkpoint
    model = load_model(hparams)
    model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
    # taco_model.cuda().eval().half()
    model.to(device).eval()
    return model


def gen_speaker_id_dict(hparams):
    speaker_id_dict = dict()
    with open(hparams.speaker_id_file, 'r') as fi:
            speaker_id_dict = json.load(fi)
        # for line in fi:
        #     speaker_name, speaker_id = line.strip().split()
        #     speaker_id_dict[speaker_name] = int(speaker_id)
    return speaker_id_dict


class Synthesizer():
    def __init__(self, model_path, out_dir, text_file, sil_file, use_griffin_lim, gen_wavenet_fea, hparams):
        self.out_dir = out_dir
        self.text_file = text_file
        self.sil_file = sil_file
        self.use_griffin_lim = use_griffin_lim
        self.gen_wavenet_fea = gen_wavenet_fea
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

        if hparams.is_multi_speakers and not hparams.use_pretrained_spkemb:
            self.speaker_id_dict = gen_speaker_id_dict(hparams)

        self.out_png_dir = os.path.join(self.out_dir, 'png')
        os.makedirs(self.out_png_dir, exist_ok=True)
        if self.use_griffin_lim:
            self.out_wav_dir = os.path.join(self.out_dir, 'wav')
            os.makedirs(self.out_wav_dir, exist_ok=True)
        if self.gen_wavenet_fea:
            self.out_mel_dir = os.path.join(self.out_dir, 'mel')
            os.makedirs(self.out_mel_dir, exist_ok=True)

    def get_mel_gt(self, wavname):
        hparams = self.hparams
        if not hparams.load_mel:
            if hparams.use_hdf5:
                with h5py.File(hparams.hdf5_file, 'r') as h5:
                    data = h5[wavname][:]
            else:
                filename = os.path.join(hparams.wav_dir, wavname + '.wav')
                sr_t, audio = wavread(filename)
                assert sr_t == hparams.sample_rate
            audio_norm = audio / hparams.max_wav_value
            wav = self.audio_class._preemphasize(audio_norm)
            melspec = self.audio_class.melspectrogram(wav, clip_norm=True)
            melspec = torch.FloatTensor(melspec.astype(np.float32))
        else:
            if hparams.use_zip:
                with zipfile.ZipFile(hparams.zip_path, 'r') as f:
                    data = f.read(wavname)
                    melspec = np.load(io.BytesIO(data))
                melspec = torch.FloatTensor(melspec.astype(np.float32))
            elif hparams.use_hdf5:
                with h5py.File(hparams.hdf5_file, 'r') as h5:
                    melspec = h5[wavname][:]
                melspec = torch.FloatTensor(melspec.astype(np.float32))
            else:
                filename = os.path.join(hparams.wav_dir, wavname + '.npy')
                melspec = torch.from_numpy(np.load(filename))
        melspec = torch.unsqueeze(melspec, 0)
        return melspec

    def get_inputs(self, meta_data):
        hparams = self.hparams
        # Prepare text input
        # filename = meta_data['n']
        # filename = os.path.splitext(os.path.basename(filename))[0]
        filename = meta_data[0].strip().split('|')[0]
        print(meta_data[0].strip().split('|')[-1])
        print(meta_data[0].strip().split('|')[1])
        sequence = np.array(self.text_to_sequence(meta_data[0].strip().split('|')[-1], ['english_cleaners']))   # [None, :]
        # sequence = torch.autograd.Variable(
        #     torch.from_numpy(sequence)).cuda().long()
        print(sequence)
        sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).to(device).long()

        if hparams.is_multi_speakers:
            if hparams.use_pretrained_spkemb:
                ref_file = meta_data['r']
                spk_embedding = np.array(np.load(ref_file))
                spk_embedding = torch.autograd.Variable(torch.from_numpy(spk_embedding)).to(device).float()
                inputs = (sequence, spk_embedding)
            else:
                speaker_name = filename.split('_')[0]
                speaker_id = self.speaker_id_dict[speaker_name]
                speaker_id = np.array([speaker_id])
                # speaker_id = torch.autograd.Variable(
                #     torch.from_numpy(speaker_id)).cuda().long()
                speaker_id = torch.autograd.Variable(
                    torch.from_numpy(speaker_id)).to(device).long()
                inputs = (sequence, speaker_id)

        if hparams.is_multi_styles:
            style_id = np.array([int(meta_data[0].strip().split('|')[1])])
            style_id = torch.autograd.Variable(
                    torch.from_numpy(style_id)).to(device).long()
            inputs = (sequence, style_id)

        elif hparams.use_vqvae:
            ref_file = meta_data['r']
            spk_ref = self.get_mel_gt(ref_file)
            inputs = (sequence, spk_ref)
        else:
            inputs = (sequence)

        return inputs, filename

    def gen_mel(self, meta_data):
        inputs, filename = self.get_inputs(meta_data)
        speaker_id = None
        style_id = None
        spk_embedding = None
        spk_ref = None
        if self.hparams.is_multi_speakers:
            if self.hparams.use_pretrained_spkemb:
                sequence, spk_embedding = inputs
            else:
                sequence, speaker_id = inputs
        elif hparams.use_vqvae:
            sequence, spk_ref = inputs
        else:
            sequence = inputs

        if self.hparams.is_multi_styles:
            sequence, style_id = inputs

        # Decode text input and plot results
        with torch.no_grad():
            mel_outputs, gate_outputs, att_ws = self.model.inference(sequence, self.hparams, 
                                                                     spk_id=speaker_id, style_id=style_id, spemb=spk_embedding, spk_ref=spk_ref)
            
            duration_list = DurationCalculator._calculate_duration(att_ws)
            print('att_ws.shape=',att_ws.shape)
            print('duration=',duration_list)
            print('duration_sum=',torch.sum(duration_list))
            print('focus_rete=',DurationCalculator._calculate_focus_rete(att_ws))
            # print(mel_outputs.shape) # (length, dim)

            mel_outputs = mel_outputs.transpose(0, 1).float().data.cpu().numpy()  # (dim, length)
            mel_outputs_with_duration = get_duration_matrix(char_text_dir=self.text_file, duration_tensor=duration_list, save_mode='phone').transpose(0, 1).float().data.cpu().numpy()
            gate_outputs = gate_outputs.float().data.cpu().numpy()
            att_ws = att_ws.float().data.cpu().numpy()

        image_path = os.path.join(self.out_png_dir, "{}_att.png".format(filename))
        _plot_and_save(att_ws, image_path)

        image_path = os.path.join(self.out_png_dir, "{}_spec_stop.png".format(filename))
        fig, axes = plt.subplots(3, 1, figsize=(8,8))
        axes[0].imshow(mel_outputs, aspect='auto', origin='bottom', 
                       interpolation='none')
        axes[1].imshow(mel_outputs_with_duration, aspect='auto', origin='bottom', 
                       interpolation='none')
        axes[2].scatter(range(len(gate_outputs)), gate_outputs, alpha=0.5,
                        color='red', marker='.', s=5, label='predicted')
        plt.savefig(image_path, format='png')
        plt.close()

        return mel_outputs, filename

    def gen_wav_griffin_lim(self, mel_outputs, filename):
        grf_wav = self.audio_class.inv_mel_spectrogram(mel_outputs)
        grf_wav = self.audio_class.inv_preemphasize(grf_wav)
        wav_path = os.path.join(self.out_wav_dir, "{}-gl.wav".format(filename))
        self.audio_class.save_wav(grf_wav, wav_path)

    def gen_wavenet_feature(self, mel_outputs, filename, add_end_sil=True):
        # denormalize
        mel = self.audio_class._denormalize(mel_outputs)
        # normalize to 0-1
        mel = np.clip(((mel - self.audio_class.hparams.min_level_db) / (-self.audio_class.hparams.min_level_db)), 0, 1)

        mel = mel.T.astype(np.float32)

        frame_size = 200
        SILSEG = 0.3
        SAMPLING = 16000
        sil_samples = int(SILSEG * SAMPLING)
        sil_frames = int(sil_samples / frame_size)
        sil_data, _ = soundfile.read(self.sil_file)
        sil_data = sil_data[:sil_samples]

        sil_mel_spec, _ = self.audio_class._magnitude_spectrogram(sil_data, clip_norm=True)
        sil_mel_spec = (sil_mel_spec + 4.0) / 8.0

        pad_mel_data = np.concatenate((sil_mel_spec[:sil_frames], mel),
                                       axis=0)
        if add_end_sil:
            pad_mel_data = np.concatenate((pad_mel_data, sil_mel_spec[:sil_frames]),
                                           axis=0)
        out_mel_file = os.path.join(self.out_mel_dir, '{}-wn.mel'.format(filename))
        save_htk_data(pad_mel_data, out_mel_file)


    def inference_f(self):
        # print(meta_data['n'])
        meta_data = _read_meta_yyh(self.text_file)
        mel_outputs, filename = self.gen_mel(meta_data)
        print('my_mel_outputs=',mel_outputs)
        print('my_mel_outputs_max=',np.max(mel_outputs))
        print('my_mel_outputs_min=',np.min(mel_outputs))
        mel_outputs = np.load(r'../out_0.npy').transpose(1,0)
        mel_outputs = mel_outputs*8.0-4.0
        print('his_mel_outputs=',mel_outputs)
        print('his_mel_outputs_max=',np.max(mel_outputs))
        print('his_mel_outputs_min=',np.min(mel_outputs))
        if self.use_griffin_lim:
            self.gen_wav_griffin_lim(mel_outputs, filename)
        if self.gen_wavenet_fea:
            self.gen_wavenet_feature(mel_outputs, filename)
        return filename

    def inference(self):
        all_meta_data = _read_meta(self.text_file, hparams.meta_format)
        list(map(self.inference_f, all_meta_data))


if __name__ == '__main__':
    def str2bool(s):
        s = s.lower()
        assert s in ["true", "false"]
        return {'t': True, 'f': False}[s[0]]

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--model_path', type=str, default="checkpoint/checkpoint_60000_smma",
                        help='transformer checkpoint path')
    parser.add_argument('-t', '--text_file', type=str, default="test_txt.txt",
                        help='text file')
    parser.add_argument('-o', '--out_dir', type=str, default="gen_wav",
                        required=False, help='output dir')
    parser.add_argument('-s', '--sil_file', type=str, default="utils/Seed_16k.wav",
                        required=False, help='silence audio')
    parser.add_argument('--use_griffin_lim', type=str2bool, default=True,
                        help='whether generate wav using grifflin lim')
    parser.add_argument('--gen_wavenet_fea', type=str2bool, default=False,
                        help='whether generate wavenet feature')
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
                              args.use_griffin_lim, args.gen_wavenet_fea, hparams)
    synthesizer.inference_f()
