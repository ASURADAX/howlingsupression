"""
1. write a function with arguments named source_path and target_path.
2. read wav files stored in source_path.
3. split wav files and stored them in target_path.
"""

import os
import torchaudio
import soundfile as sf

def split_wav_audio_files(source_path, target_path):
    # get the list of wav files in the source path
    wav_files = os.listdir(source_path)
    
    # loop over each wav file
    for file in wav_files:
        # load the audio file
        signal, sr = torchaudio.load(os.path.join(source_path, file))

        # split the audio file
        split_at_timestamp  = 4
        split_at_frame = split_at_timestamp * sr  
        # 完整长度的分段数量
        num_splits = (int)(signal.size(1) / split_at_frame)

        # loop over each split of wav file
        start_frame = 0
        n = 0
        for num in range(num_splits):
            # get the split audio
            split_signal = signal[:,start_frame:start_frame + split_at_frame]
            
            # save the split audio
            sf.write(os.path.join(target_path, '{}_{}.wav'.format(file.split('.')[0], num)), split_signal.numpy().transpose(), sr)
            
            start_frame += split_at_frame
            n = num + 1
        # 处理剩余部分
        if(start_frame<signal.size(1)):
            split_signal = signal[:,start_frame:]
            sf.write(os.path.join(target_path, '{}_{}.wav'.format(file.split('.')[0], n)), split_signal.numpy().transpose(), sr)

        

if __name__ == "__main__":
    source_path = './wav'
    target_path = './split'
    split_wav_audio_files(source_path,target_path)

