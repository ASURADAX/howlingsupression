import pyaudio # 收集声音
import numpy as np # 处理声音数据
import matplotlib.pyplot as plt # 作图
import matplotlib as mpl 
import librosa.core as lc
from scipy import signal

END = False
No_colobar=True
# 按键中断: 按下按键q执行该函数
def on_press(event):
    global stream, p, END
    if event.key == 'q':
        plt.close()
        stream.stop_stream()
        stream.close()
        p.terminate()
        END = True

# 输入音频参数设置
CHUNK = 1024 * 8
FORMAT = pyaudio.paInt16
CHANNEL = 1
RATE = 44100
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNEL, rate=RATE,\
    input=True, frames_per_buffer=CHUNK)
player = p.open(format=pyaudio.paInt16, channels=1, rate=RATE,\
    output=True, frames_per_buffer=CHUNK)

# 作图的设置

mpl.rcParams['toolbar'] = 'None'
fig=plt.figure()
ax = fig.add_subplot(211)
ax1= fig.add_subplot(212)
#plt.subplots_adjust(left=0.1, top=0.9,right=0.9,bottom=0.1)
plt.subplot(212)
ax1.set_yscale('log')
plt.title('STFT Magnitude')
#ax1.set_xlabel('Time [sec]', fontsize=12)
#ax1.set_ylabel('Frequency [Hz]', fontsize=12)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.tight_layout()
fig.canvas.mpl_connect('key_press_event', on_press)
#plt.subplots_adjust(left=0.001, top=0.999,right=0.999,bottom=0.001)
plt.get_current_fig_manager().set_window_title('Wave')
x = np.arange(0,CHUNK)
#line, = ax.plot(x, np.random.rand(CHUNK), color='#C04851')
#设置坐标轴视图限制
ax.set_xlim(0,CHUNK-1)
ax.set_ylim(-2**15,2**15)
#隐藏坐标轴
#plt.axis('off')
#打开交互模式
plt.ion()
plt.show()

# 程序主体
while END==False:
    indata = stream.read(CHUNK, exception_on_overflow=False)
    data0 = np.frombuffer(indata, dtype=np.int16)
    fre, ts, zxx = signal.stft(data0,RATE,nperseg=512)
    ax1.set_ylim([fre[1], fre[-1]])
    img = ax1.pcolormesh(ts, fre, np.abs(zxx))
    #line.set_ydata(data)
    _, stft_data = signal.istft(zxx, RATE)
    ax.plot(x, data0, x, stft_data)
    ax.legend(['Carrier', 'Filtered via STFT'])
    player.write(np.fromstring(stft_data,dtype=np.int16))
    if No_colobar:
        No_colobar=False
        fig.colorbar(img,ax=ax1)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)

stream.stop_stream()
stream.close()
p.terminate()