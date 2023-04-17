import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from dnn.utils import get_spectrograms
import dnn.hyperparams as hp
import librosa
class PrepareDataset(Dataset):
    """wav to mel dataset."""

    def __init__(self, wav_dir):
        self.wav_dir = wav_dir
        self.wav_files = list(map(lambda x: os.path.join(wav_dir, x), os.listdir(wav_dir)))
        
    def __len__(self):
        return len(self.wav_files)

    def load_wav(self, filename):
        return librosa.load(filename, sr=hp.sample_rate)

    def __getitem__(self, idx):
        wav_file = self.wav_files[idx]
        mel, mag = get_spectrograms(wav_file)

        np.save(wav_file[:-4] + '.pt', mel)
        np.save(wav_file[:-4] + '.mag', mag)

        sample = {'mel':mel, 'mag': mag}

        return sample
    
if __name__ == '__main__':
    wav_path = './howling/wav'
    dataset = PrepareDataset(wav_path)
    dataloader = DataLoader(dataset, batch_size=1, drop_last=False, num_workers=8)
    from tqdm import tqdm
    pbar = tqdm(dataloader)
    for d in pbar:
        pass
