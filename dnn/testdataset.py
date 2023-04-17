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

class TestDatasets(Dataset):
    def __init__(self, source_dir, target_dir):
        """
        Args:
            source_dir (string): Directory with the source data.
            target_dir (string): Directory with the target data.
        the naming convention:
            source_dir:
                h_cv_1.pt.npy
            target_dir:
                cv_1.wav
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
                target_file = os.path.join(self.target_dir, f"cv_{index}.wav")
                if os.path.exists(target_file):
                    file_list.append((file_name, index))
        return file_list
    
    def __getitem__(self, idx):
        file_name, index = self.file_list[idx]
        source_file = os.path.join(self.source_dir, file_name)
        target_file = os.path.join(self.target_dir, f"cv_{index}.wav")

        source_mel = np.load(source_file)
        target_wav, sr = self.load_wav(target_file)
        # add the <Go> frame
        source_mel_input = np.concatenate([np.zeros([1,hp.num_mels], np.float32), source_mel[:-1,:]], axis=0)

        source_pos_mel = np.arange(1, source_mel.shape[0] + 1)

        sample = {'source_mel': source_mel, 'source_mel_input':source_mel_input, 'source_pos_mel':source_pos_mel, 'target_wav':target_wav}
        
        return sample

def collate_fn_transformer_test(batch):
    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.Mapping):
        source_mel = [d['source_mel'] for d in batch]
        source_mel_input = [d['source_mel_input'] for d in batch]
        source_pos_mel = [d['source_pos_mel'] for d in batch]
        target_wav = [d['target_wav'] for d in batch]
        # PAD sequences with largest length of the batch
        source_mel = _pad_mel(source_mel)
        source_mel_input = _pad_mel(source_mel_input)
        source_pos_mel = _prepare_data(source_pos_mel).astype(np.int32)
        target_wav = _pad_wav(target_wav)
        return t.FloatTensor(source_mel),t.FloatTensor(source_mel_input),t.LongTensor(source_pos_mel),t.FloatTensor(target_wav)
    
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

def get_testset():
    source_dir = './howl_1/pt'
    target_dir = './wav_1/wav'
    # source_dir = './dnn/samples/train_data/source'
    # target_dir = './dnn/samples/train_data/target'
    return TestDatasets(source_dir,target_dir)
    #return LJDatasets(os.path.join(hp.data_path,'metadata.csv'), os.path.join(hp.data_path,'wavs'))

def _pad_mel(inputs):
    _pad = 0
    def _pad_one(x, max_len):
        mel_len = x.shape[0]
        return np.pad(x, [[0,max_len - mel_len],[0,0]], mode='constant', constant_values=_pad)
    max_len = max((x.shape[0] for x in inputs))
    return np.stack([_pad_one(x, max_len) for x in inputs])

def _pad_wav(inputs):
    _pad = 0
    def _pad_one(x, max_len):
        wav_len = x.shape[0]
        return np.pad(x,(0, max_len - wav_len), mode='constant', constant_values=_pad)
    max_len = max((x.shape[0] for x in inputs))
    return np.stack([_pad_one(x, max_len) for x in inputs])
    #return np.stack([t.from_numpy(_pad_one(x, max_len)).float() for x in inputs])


if __name__ == '__main__':
    dataset = get_testset()
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn_transformer_test,drop_last=False, num_workers=1)
    #from tqdm import tqdm
    pbar = dataloader
    for i, data in enumerate(pbar):
        source_mel, source_mel_input, source_pos_mel, target_wav = data
        #print(target_wav.shape)
        #if isinstance(target_wav,t.Tensor):
        #    print("TRUE")