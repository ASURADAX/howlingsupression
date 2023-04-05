import hyperparams as hp
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import librosa
import numpy as np
#from text import text_to_sequence
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
        the naming convention:
            source_dir:
                h_cv_1.pt.npy
            target_dir:
                cv_1.pt.npy
        """
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.file_list = self._get_file_list()

    def load_wav(self, filename):
        return librosa.load(filename, sr=hp.sr)

    def __len__(self):
        return len(self.file_list)

    def _get_file_list(self):
        # Get a list of file names that match the naming convention
        file_list = []
        for file_name in os.listdir(self.source_dir):
            if file_name.startswith("h_cv_") and file_name.endswith(".pt.npy"):
                index = file_name.split("_")[-1].split(".")[0]
                target_file = os.path.join(self.target_dir, f"cv_{index}.pt.npy")
                if os.path.exists(target_file):
                    file_list.append((file_name, index))
        return file_list

    def __getitem__(self, idx):
        file_name, index = self.file_list[idx]
        source_file = os.path.join(self.source_dir, file_name)
        target_file = os.path.join(self.target_dir, f"cv_{index}.pt.npy")

        source_mel = np.load(source_file)
        target_mel = np.load(target_file)
        # add the <Go> frame
        source_mel_input = np.concatenate([np.zeros([1,hp.num_mels], np.float32), source_mel[:-1,:]], axis=0)
        target_mel_input = np.concatenate([np.zeros([1,hp.num_mels], np.float32), target_mel[:-1,:]], axis=0)
        
        source_pos_mel = np.arange(1, source_mel.shape[0] + 1)
        target_pos_mel = np.arange(1, target_mel.shape[0] + 1)

        sample = {'source_mel': source_mel, 'target_mel': target_mel, 'source_mel_input':source_mel_input, 'target_mel_input':target_mel_input, 'source_pos_mel':source_pos_mel, 'target_pos_mel':target_pos_mel}

        return sample


class PostDatasets(Dataset):
    """mel to mag dataset."""

    def __init__(self, mel_dir, mag_dir):
        """
        Args:
            mel_dir (string): Directory with all the mels.
            mag_dir (string): Directory with all the mags.
        the naming convention:
            mel_dir:
                cv_1.pt.npy
            target_dir:
                cv_1.mag.npy
        """
        self.mel_dir=mel_dir
        self.mag_dir=mag_dir
        self.file_list = self._get_file_list()

    def _get_file_list(self):
        # Get a list of file names that match the naming convention
        file_list = []
        for file_name in os.listdir(self.mel_dir):
            if file_name.startswith("cv_") and file_name.endswith(".pt.npy"):
                index = file_name.split("_")[-1].split(".")[0]
                target_file = os.path.join(self.mag_dir, f"cv_{index}.mag.npy")
                if os.path.exists(target_file):
                    file_list.append((file_name, index))
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name, index = self.file_list[idx]
        mel_file = os.path.join(self.mel_dir, file_name)
        mag_file = os.path.join(self.mag_dir, f"cv_{index}.mag.npy")
        mel = np.load(mel_file)
        mag = np.load(mag_file)
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
    source_dir = './howling/pt'
    target_dir = './wav/pt'
    # source_dir = './dnn/samples/train_data/source'
    # target_dir = './dnn/samples/train_data/target'
    return HowlingDatasets(source_dir,target_dir)
    #return LJDatasets(os.path.join(hp.data_path,'metadata.csv'), os.path.join(hp.data_path,'wavs'))

def get_post_dataset():
    mel_dir = './wav/pt'
    mag_dir = './wav/mag'
    return PostDatasets(mel_dir=mel_dir,mag_dir=mag_dir)
    #return PostDatasets(os.path.join(hp.data_path,'metadata.csv'), os.path.join(hp.data_path,'wavs'))

def _pad_mel(inputs):
    _pad = 0
    def _pad_one(x, max_len):
        mel_len = x.shape[0]
        return np.pad(x, [[0,max_len - mel_len],[0,0]], mode='constant', constant_values=_pad)
    max_len = max((x.shape[0] for x in inputs))
    return np.stack([_pad_one(x, max_len) for x in inputs])

if __name__ == '__main__':
    dataset = get_post_dataset()
    dataloader = DataLoader(dataset, batch_size=1, drop_last=False, num_workers=1)
    #from tqdm import tqdm
    pbar = dataloader
    for d in pbar:
        pass