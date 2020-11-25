import os, io
import random
import json
import h5py
import numpy as np
import zipfile
import torch
import torch.utils.data
from torch.utils.data import sampler
from .audio import Audio
from scipy.io.wavfile import read as wavread
from espnet2.tts.duration_calculator import DurationCalculator
from utils.utils import AverageMeter

class TextMelLoader_refine(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, metadata_file_path, hparams):
        # self.audiopaths_and_text = load_filepaths_and_text(metadata_file_path)
       
        self._metadata = _read_meta_yyh(metadata_file_path)

        self.mel_dir = hparams.mel_dir

        if hparams.use_phone:
            from text.phones import Phones
            self.phone_class = Phones(hparams.phone_set_file)
            self.text_to_sequence = self.phone_class.text_to_sequence
        else:
            from text import text_to_sequence
            self.text_to_sequence = text_to_sequence
        
        self.text_cleaners = hparams.text_cleaners
        self.is_multi_speakers = hparams.is_multi_speakers
        self.is_multi_styles = hparams.is_multi_styles

        # random.seed(1234)
        random.shuffle(self._metadata)
        self.utt_list = []
        self.utt_data = {}
        self.utt_mels = []
        for i in range(len(self._metadata)):
            item = self._metadata[i]
            self.utt_list.append({"mel_frame": int(item.strip().split('|')[-1])})
            self.utt_data[str(i)] = {"text":torch.IntTensor(self.text_to_sequence(item.strip().split('|')[5] , self.text_cleaners)),
                                    "mel":torch.from_numpy(np.load(self.mel_dir + 'out_' + item.strip().split('|')[0] + '.npy'))*8.0-4.0,
                                    "dur":DurationCalculator._calculate_duration(torch.from_numpy(np.load(self.mel_dir + 'encdec_' + item.strip().split('|')[0] + '.npy'))),
                                    "speaker":int(item.strip().split('|')[4]) if self.is_multi_speakers else None,
                                    "style":int(item.strip().split('|')[2])if self.is_multi_styles else None}
            # self.utt_list.append({"mel_frame": int(np.load(self.mel_dir + 'out_' + item.strip().split('|')[0] + '.npy').shape[0])})
            if hparams.is_refine_style:
                self.utt_mels.append((torch.from_numpy(np.load(self.mel_dir + 'out_' + item.strip().split('|')[0] + '.npy'))*8.0-4.0).unsqueeze(0))        
    def get_mel_text_pair(self, meta_data):
        text = meta_data["text"]
        mel = meta_data["mel"]
        dur = meta_data["dur"]
        
        ret_pair = [text, mel, dur]

        if self.is_multi_speakers:
            ret_pair.append(meta_data["speaker"])

        if self.is_multi_styles:
            ret_pair.append(meta_data["style"])

        return ret_pair

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.utt_data[str(index)])

    def __len__(self):
        return len(self.utt_data)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, hparams):
        self.n_frames_per_step = hparams.reduction_factor
        self.is_multi_speakers = hparams.is_multi_speakers
        self.is_multi_styles = hparams.is_multi_styles
        self.num_mels = hparams.num_mels

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized, dur, speaker_id/style_id]   ###mel_normalized.size=(mel_nums,frames) but (frames, mel_nums) from loaded 
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        duration_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        duration_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            dur = batch[ids_sorted_decreasing[i]][2]
            text_padded[i, :text.size(0)] = text
            duration_padded[i, :dur.size(0)] = dur

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(1)
        assert num_mels == self.num_mels
        max_target_len = max([x[1].size(0) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), max_target_len, num_mels)
        mel_padded.zero_()

        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :mel.size(0), :] = mel
            output_lengths[i] = mel.size(0)

        res = [text_padded, input_lengths, mel_padded, output_lengths, duration_padded, input_lengths]

        idx = 3
        if self.is_multi_speakers:
            speaker_ids = torch.LongTensor(len(batch))
            for i in range(len(ids_sorted_decreasing)):
                speaker_ids[i] = batch[ids_sorted_decreasing[i]][idx]
            res.append(speaker_ids)
            idx += 1

        if self.is_multi_styles:
            style_ids = torch.LongTensor(len(batch))#(B,)
            for i in range(len(ids_sorted_decreasing)):
                style_ids[i] = batch[ids_sorted_decreasing[i]][idx]
            res.append(style_ids)
            idx +=1

        return res


class DynamicBatchSampler(sampler.Sampler):
    """Extension of Sampler that will do the following:
        1.  Change the batch size (essentially number of sequences) 
            in a batch to ensure that the total number of frames are less
            than a certain threshold. 
        2.  Make sure the padding efficiency in the batch is high.
    """

    def __init__(self, sampler, frames_threshold):
        """
        @sampler: will mostly be an instance of DistributedSampler.
        Though it should work with any sampler.
        @frames_threshold: maximum area of the batch
        """
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        # indices = [(idx1, #frames1), (idx2, #frames2), ...]
        # batches = [[idx1, idx2, idx3], [idx4], [idx5, idx6], ...]
        indices, batches = list(), list()
        # the dataset to which these indices are pointing to
        dataset = self.sampler.dataset
        # get all the indices and corresponding durations from 
        # the sampler
        for idx in self.sampler:
            indices.append((idx, dataset.utt_list[idx]["mel_frame"]))
        # sort the indices according to duration
        indices.sort(key=lambda elem : int(elem[1]))
        # make sure that you will be able to serve all the utterances
        assert indices[-1][1] < self.frames_threshold, ("Won't be able"
            "to serve all sequences. frames_threshold={} while longest"
            " utterance has {} frames").format(self.frames_threshold, 
                                                indices[-1][1])
        # start clubbing the utterances together
        batch = list()
        batch_frames, batch_area = 0, 0
        average_meter = AverageMeter('Padding Efficiency')
        for idx, frames in indices:
            # consider adding this utterance to the current batch
            if batch_frames + frames <= self.frames_threshold:
                # can add to the batch
                batch.append(idx)
                batch_frames += frames
                batch_area = frames * len(batch)
                # print('idx=',idx)
                # print('frames=',frames)
                # print('batch_area=',batch_area)
            else:
                # log the stats and add previous batch to batches
                average_meter.add(batch_frames, batch_area)
                batches.append(batch)
                # make a new one
                batch = list([idx])
                batch_frames, batch_area = frames, frames
        # don't need the 'indices' any more
        del indices
        self.batches = batches
        # average_meter.display_results()
    
    def __iter__(self):
        # shuffle on a batch level
        # random.seed(1234)
        random.shuffle(self.batches)
        return iter(self.batches)
    
    def __len__(self):
        return len(self.batches)

def _read_meta_yyh(meta_file):
    File_final = open(meta_file,encoding='utf-8')
    uttline_temp = File_final.readlines()
    return uttline_temp