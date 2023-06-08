import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# 读取wav文件
fs, data = wavfile.read('./howling/wav/h_cv_0.wav')
data = data / (2.0 ** 15)    # 归一化

# 创建一行两个子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# 绘制时域图
ax1.plot(np.arange(len(data)) / float(fs), data)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
ax1.set_title('Time Domain')

# 绘制时频图
nfft = 1024
window = np.hamming(nfft)
im = ax2.specgram(data, NFFT=nfft, Fs=fs, window=window, noverlap=nfft/2)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Frequency (Hz)')
ax2.set_title('Spectrogram')
cb = plt.colorbar(im[3], ax=ax2)

# 调整子图间距
plt.subplots_adjust(wspace=0.3)

plt.show()