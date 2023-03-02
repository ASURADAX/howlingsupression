import hyperparams as hp
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import librosa
import numpy as np
from text import text_to_sequence
import collections
from scipy import signal
import torch as t
import math

class HowlingDatasets(Dataset):
    def __init__(self, source_dir, target_dir):
        """
        Args:
            source_dir (string): Directory with the source data.
            target_dir (string): Directory with the target data.

        """
        self.source_dir = source_dir
        self.target_dir = target_dir
        # 文件路径名
        self.source_files = list(map(lambda x: os.path.join(source_dir, x), os.listdir(source_dir)))
        self.target_files = list(map(lambda x: os.path.join(target_dir, x), os.listdir(target_dir)))

    def load_wav(self, filename):
        return librosa.load(filename, sr=hp.sr)

    def __len__(self):
        assert len(self.source_files) == len(self.target_files)
        return len(self.source_files)

    def __getitem__(self, idx):
        source_file = self.source_files[idx]
        target_file = self.target_files[idx]

        source_mel = np.load(source_file)
        target_mel = np.load(target_file)
        # add the <Go> frame
        source_mel_input = np.concatenate([np.zeros([1,hp.num_mels], np.float32), source_mel[:-1,:]], axis=0)
        target_mel_input = np.concatenate([np.zeros([1,hp.num_mels], np.float32), target_mel[:-1,:]], axis=0)
        
        source_pos_mel = np.arange(1, source_mel.shape[0] + 1)
        target_pos_mel = np.arange(1, target_mel.shape[0] + 1)

        sample = {'source_mel': source_mel, 'target_mel': target_mel, 'source_mel_input':source_mel_input, 'target_mel_input':target_mel_input, 'source_pos_mel':source_pos_mel, 'target_pos_mel':target_pos_mel}

        return sample

class LJDatasets(Dataset):
    """LJSpeech dataset."""

    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.

        """
        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir

    def load_wav(self, filename):
        return librosa.load(filename, sr=hp.sample_rate)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        wav_name = os.path.join(self.root_dir, self.landmarks_frame.ix[idx, 0]) + '.wav'
        text = self.landmarks_frame.ix[idx, 1]

        text = np.asarray(text_to_sequence(text, [hp.cleaners]), dtype=np.int32)
        mel = np.load(wav_name[:-4] + '.pt.npy')
        mel_input = np.concatenate([np.zeros([1,hp.num_mels], np.float32), mel[:-1,:]], axis=0)
        text_length = len(text)
        pos_text = np.arange(1, text_length + 1)
        pos_mel = np.arange(1, mel.shape[0] + 1)

        sample = {'text': text, 'mel': mel, 'text_length':text_length, 'mel_input':mel_input, 'pos_mel':pos_mel, 'pos_text':pos_text}

        return sample
    
class PostDatasets(Dataset):
    """LJSpeech dataset."""

    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.

        """
        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        wav_name = os.path.join(self.root_dir, self.landmarks_frame.ix[idx, 0]) + '.wav'
        mel = np.load(wav_name[:-4] + '.pt.npy')
        mag = np.load(wav_name[:-4] + '.mag.npy')
        sample = {'mel':mel, 'mag':mag}

        return sample
    
def collate_fn_transformer(batch):

    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.Mapping):

        source_mel = [d['source_mel'] for d in batch]
        target_mel = [d['target_mel'] for d in batch]
        source_mel_input = [d['source_mel_input'] for d in batch]
        target_mel_input = [d['target_mel_input'] for d in batch]
        source_pos_mel = [d['source_pos_mel'] for d in batch]
        target_pos_mel = [d['target_pos_mel'] for d in batch]

        #text = [d['text'] for d in batch]
        #mel = [d['mel'] for d in batch]
        #mel_input = [d['mel_input'] for d in batch]
        #text_length = [d['text_length'] for d in batch]
        #pos_mel = [d['pos_mel'] for d in batch]
        #pos_text= [d['pos_text'] for d in batch]
        
        # 按字符串长度排序
        #text = [i for i,_ in sorted(zip(text, text_length), key=lambda x: x[1], reverse=True)]
        #mel = [i for i, _ in sorted(zip(mel, text_length), key=lambda x: x[1], reverse=True)]
        #mel_input = [i for i, _ in sorted(zip(mel_input, text_length), key=lambda x: x[1], reverse=True)]
        #pos_text = [i for i, _ in sorted(zip(pos_text, text_length), key=lambda x: x[1], reverse=True)]
        #pos_mel = [i for i, _ in sorted(zip(pos_mel, text_length), key=lambda x: x[1], reverse=True)]
        #text_length = sorted(text_length, reverse=True)
        # PAD sequences with largest length of the batch
        source_mel = _pad_mel(source_mel)
        target_mel = _pad_mel(target_mel)
        source_mel_input = _pad_mel(source_mel_input)
        target_mel_input = _pad_mel(target_mel_input)
        source_pos_mel = _prepare_data(source_pos_mel).astype(np.int32)
        target_pos_mel = _prepare_data(target_pos_mel).astype(np.int32)
        #text = _prepare_data(text).astype(np.int32)
        #mel = _pad_mel(mel)
        #mel_input = _pad_mel(mel_input)
        #pos_mel = _prepare_data(pos_mel).astype(np.int32)
        #pos_text = _prepare_data(pos_text).astype(np.int32)

        return t.FloatTensor(source_mel),t.FloatTensor(target_mel),t.FloatTensor(source_mel_input),t.FloatTensor(target_mel_input),t.LongTensor(source_pos_mel),t.LongTensor(target_pos_mel)
        #return t.LongTensor(text), t.FloatTensor(mel), t.FloatTensor(mel_input), t.LongTensor(pos_text), t.LongTensor(pos_mel), t.LongTensor(text_length)

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))
    
def collate_fn_postnet(batch):

    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.Mapping):

        mel = [d['mel'] for d in batch]
        mag = [d['mag'] for d in batch]
        
        # PAD sequences with largest length of the batch
        mel = _pad_mel(mel)
        mag = _pad_mel(mag)

        return t.FloatTensor(mel), t.FloatTensor(mag)

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))

def _pad_data(x, length):
    _pad = 0
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)

def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])

def _pad_per_step(inputs):
    timesteps = inputs.shape[-1]
    return np.pad(inputs, [[0,0],[0,0],[0, hp.outputs_per_step - (timesteps % hp.outputs_per_step)]], mode='constant', constant_values=0.0)

def get_param_size(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params

def get_dataset():
    return LJDatasets(os.path.join(hp.data_path,'metadata.csv'), os.path.join(hp.data_path,'wavs'))

def get_post_dataset():
    return PostDatasets(os.path.join(hp.data_path,'metadata.csv'), os.path.join(hp.data_path,'wavs'))

def _pad_mel(inputs):
    _pad = 0
    def _pad_one(x, max_len):
        mel_len = x.shape[0]
        return np.pad(x, [[0,max_len - mel_len],[0,0]], mode='constant', constant_values=_pad)
    max_len = max((x.shape[0] for x in inputs))
    return np.stack([_pad_one(x, max_len) for x in inputs])

