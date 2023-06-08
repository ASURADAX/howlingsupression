import numpy as np
import matplotlib.pyplot as plt

source_file = './wav/pt/cv_0.pt.npy'
source_mel = np.load(source_file)
print(source_mel.shape)

# 绘制梅尔频谱图
plt.imshow(source_mel.T, aspect='auto', origin='lower', cmap='coolwarm', interpolation='nearest', extent=[0, 5, 0, 8000])

# 设置图形标题和横纵坐标标签
plt.title('Mel Spectrogram')
plt.xlabel('Time (Seconds)')
plt.ylabel('Frequency (Hz)')

# 添加颜色刻度栏
cb = plt.colorbar()
cb.set_label('Power (dB)')

# 显示图像
plt.show()