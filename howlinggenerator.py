from scipy import signal
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
import random
import numpy as np

sr = 22050

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
        ir_spec = torch.fft.rfft(self.IR)
        ir_mag = torch.abs(ir_spec)

        MLG = torch.mean(torch.abs(ir_mag) ** 2)
        mean_gain = 10 * torch.log10(MLG)
        reqdBLoss = target_gain - mean_gain
        factor = 0.5 ** (reqdBLoss / 6)
        self.IR = self.IR / factor

    def get_gain(self):
        return (self.gain_floor - self.gain_ceil) * random.random() + self.gain_ceil

    def howling(self, x):
        sample_len = x.size(1)
        howling_out = torch.zeros(x.size())
        conv_len = self.frame_len + self.IR.size(1) - 1
        frame_start = 0
        print(x.shape)
        for i in range(sample_len):
            cur_frame = x[:, frame_start:frame_start+self.frame_len]
            windowed_frame = self.win * cur_frame
            howling_out[:,frame_start:frame_start+self.frame_len] += windowed_frame
            ##注意windowed_frame可能是双通道的，则windowed_frame.flatten()的长度将是frame_len*2
            conv_frame=np.zeros([x.size(0),conv_len])
            for j in range(x.size(0)):
                conv_frame[j]=np.convolve(windowed_frame[j].flatten(), self.IR[0].flatten(), mode="full")                   
            #print(conv_frame.shape)
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

    def forward(self, x):
        target_gain = self.get_MSG() + 2
        self.scale_IR(target_gain)
        x = self.howling(x)
        return x

if __name__ == "__main__":
    music, sr1 = torchaudio.load("./sample/MUSIC01.wav")
    music = torchaudio.functional.resample(music, orig_freq=sr1, new_freq=sr)
    IR, sr2 = torchaudio.load("./sample/IR01.wav")
    IR = torchaudio.functional.resample(IR, orig_freq=sr2, new_freq=sr)

    transformer = HowlingTransform(IR)
    s = transformer(music)

    sf.write("./sample/s_howling_trans02.wav", s.numpy().flatten(), sr)