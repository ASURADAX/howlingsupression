from torch.utils.data import Dataset, DataLoader
import os
from pydub import AudioSegment

class PreprocessDataset(Dataset):
    """mp3 to wav and rename like cv_1.wav dataset."""

    def __init__(self, mp3_dir, wav_dir):
        self.mp3_dir = mp3_dir
        self.wav_dir = wav_dir
        self.mp3_files = list(map(lambda x: os.path.join(mp3_dir, x), os.listdir(mp3_dir)))
        
    def __len__(self):
        return len(self.mp3_files)

    def __getitem__(self, idx):
        # Load your MP3 file
        sound = AudioSegment.from_mp3(self.mp3_files[idx])
        # Export the audio to WAV format
        wav_file_path = os.path.join(self.wav_dir, 'cv_{}.wav'.format(idx))
        sound.export(wav_file_path, format="wav")
        
        sample = ""
        return sample

if __name__ == '__main__':
    wav_path = './wav'
    mp3_path = './mp3'
    dataset = PreprocessDataset(mp3_path,wav_path)
    dataloader = DataLoader(dataset, batch_size=1, drop_last=False, num_workers=1)
    from tqdm import tqdm
    pbar = tqdm(dataloader)
    num = 200  # 操作的文件数量
    i = 1
    for d in pbar:
        i = i + 1
        if(i==num):
            break