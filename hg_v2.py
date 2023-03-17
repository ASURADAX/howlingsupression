import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import stft

# 设置信号参数
fs = 22050  # 采样频率
f1 = 20  # 基频
f2 = 200  # 最高频率
sr = 22050
# 生成时间序列
t = np.arange(0, 5, 1/fs)

# 生成正弦波信号
signal = np.sin(2*np.pi*f1*t + 10*np.sin(2*np.pi*f2*t))

 # 加入高斯噪声
noise = np.random.normal(0, 0.1, len(signal))
signal += noise

# 调整频率和幅度
signal *= np.cos(2*np.pi*50*t) * np.exp(-t)
sf.write("./hg_v2.wav", signal.flatten(), sr)
# 绘制信号图像
plt.figure(figsize=(10, 5))
plt.plot(t, signal)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Chirp Signal')
plt.grid(True)

# STFT
f, t, Zxx = stft(signal, fs=fs, nperseg=256, noverlap=128)
# 绘制STFT图
plt.figure(figsize=(10, 5))
plt.pcolormesh(t, f, np.abs(Zxx), cmap='jet')
plt.title('STFT Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.ylim(0, fs/2)
plt.colorbar()


# 绘制频谱图
plt.figure(figsize=(10, 5))
f = np.fft.fftfreq(len(signal), 1/fs)  # 频率轴
S = np.fft.fft(signal)  # 傅里叶变换
plt.plot(f, np.abs(S))
plt.title('Frequency Domain')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, fs/2)
plt.grid(True)

plt.show()

