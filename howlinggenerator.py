from scipy import signal
import torch
import os
import torch.nn as nn
import torchaudio
import soundfile as sf
import random
import numpy as np
import dnn.hyperparams as hp
class HowlingTransform(nn.Module):
    def __init__(self, IR, gain_floor=1, gain_ceil=10, frame_len=128, hop_len=None):
        super().__init__()
        self.IR = IR
        self.gain_floor = gain_floor
        self.gain_ceil = gain_ceil
        self.frame_len = frame_len
        if hop_len is None:
            self.hop_len = self.frame_len // 2
        else:
            self.hop_len = hop_len
        self.win = torch.hann_window(self.frame_len)

    def get_MSG(self):
        ir_spec = torch.fft.rfft(self.IR)
        ir_mag = torch.abs(ir_spec)
        ir_phase = torch.angle(ir_spec)

        MLG = torch.mean(torch.abs(ir_mag) ** 2)
        zero_phase_index = np.where(np.logical_and(-0.1 < ir_phase, ir_phase < 0.1))
        ir_zero_phase_mag = ir_mag[zero_phase_index]
        peak_gain = torch.max(torch.abs(ir_zero_phase_mag) ** 2)
        MSG = -10 * torch.log10(peak_gain / MLG)

        return MSG

    def scale_IR(self, target_gain):
        # 对IR进行傅里叶变换，得到其频谱(ir_spec)
        ir_spec = torch.fft.rfft(self.IR)
        # 计算频谱的幅度(ir_mag)
        ir_mag = torch.abs(ir_spec)
        # 计算IR的平均幅度(MLG)
        MLG = torch.mean(torch.abs(ir_mag) ** 2)
        # 计算IR的平均增益(mean_gain)，即将平均幅度转换为分贝(dB)的值
        mean_gain = 10 * torch.log10(MLG)
        # 计算需要调整的增益(reqdBLoss)，即目标增益与平均增益之差
        reqdBLoss = target_gain - mean_gain
        # 根据需要调整的增益，计算缩放因子(factor)，使得IR的平均增益达到目标增益。
        factor = 0.5 ** (reqdBLoss / 6)
        # 将IR除以缩放因子，即可完成音量调整
        self.IR = self.IR / factor

    def get_gain(self):
        return (self.gain_floor - self.gain_ceil) * random.random() + self.gain_ceil

    def howling(self, x):
        sample_len = x.size(1)
        howling_out = torch.zeros(x.size())
        conv_len = self.frame_len + self.IR.size(1) - 1
        frame_start = 0
        #print(x.shape)
        for i in range(sample_len):
            cur_frame = x[:, frame_start:frame_start+self.frame_len]
            windowed_frame = self.win * cur_frame
            howling_out[:,frame_start:frame_start+self.frame_len] += windowed_frame
            ##注意windowed_frame可能是双通道的，则windowed_frame.flatten()的长度将是frame_len*2
            conv_frame=np.zeros([x.size(0),conv_len])
            for j in range(x.size(0)):
                conv_frame[j]=np.convolve(windowed_frame[j].flatten(), self.IR[0].flatten(), mode="full")                   
            #print(conv_frame.shape)
            # 叠加到下一帧
            frame_start = frame_start + self.hop_len
            if frame_start+conv_len < sample_len:
                x[:,frame_start:frame_start+conv_len] += conv_frame
                #assert 1!=1
            else:
                break

            x = torch.minimum(x, torch.ones(sample_len))
            x = torch.maximum(x, -torch.ones(sample_len))

        howling_out = torch.minimum(howling_out, torch.ones(sample_len))
        howling_out = torch.maximum(howling_out, -torch.ones(sample_len))
        return howling_out

    def forward(self, x, sr):
        target_gain = self.get_MSG() + 100
        self.scale_IR(target_gain)
        h = self.howling(x)
        # 调整啸叫信号频率和幅度
        t = np.arange(0, h.size(1))
        cyc = 0.5 # 控制周期数，值越大峰越多
        slope = 10 # 控制峰的陡峭程度，值越大峰越陡
        tmp = np.abs(np.cos(cyc*np.pi*t))*slope
        ttmp = np.cos(2*np.pi*50*t) * np.exp(-tmp)
        h *= ttmp
        # 叠加啸叫信号
        x += h
        return x

def add_howling(source_path, target_path):
    # get the list of wav files in the source path
    wav_files = os.listdir(source_path)
    
    # get the IR signal
    IR, sr = torchaudio.load("./sample/IR01.wav")
    IR = torchaudio.functional.resample(IR, orig_freq=sr, new_freq=hp.sr)
    
    # initialize the HowlingTransform
    transformer = HowlingTransform(IR)
    i=0
    # loop over each wav file
    for file in wav_files:
        # load the audio file
        signal, sr = torchaudio.load(os.path.join(source_path, file))
        signal = torchaudio.functional.resample(signal, orig_freq=sr, new_freq=hp.sr)
        
        # add howling to the audio file
        signal = transformer(signal,hp.sr)
        i=i+1
        print(i)
        #save the audio
        sf.write(os.path.join(target_path, 'h_{}.wav'.format(file.split('.')[0])), signal.numpy().transpose(), hp.sr)

if __name__ == "__main__":
    source_path = './wav'
    target_path = './howling'
    add_howling(source_path,target_path)
    