import pyaudio # 收集声音
import numpy as np # 处理声音数据
import matplotlib.pyplot as plt # 作图
import matplotlib as mpl 
import librosa.core as lc
from scipy import signal
import queue
import analyse

END = False
FIRST_CYCLE = True
indata = None 
outdata = None
q_len=1000
qin=queue.Queue(q_len)
qout=queue.Queue(q_len)
# 按键中断: 按下按键q执行该函数
def on_press(event):
    global stream, p, END
    if event.key == 'q':
        plt.close()
        stream.stop_stream()
        stream.close()
        p.terminate()
        END = True

# 中断函数
def callback(in_data, frame_count, time_info, status):
    global qin,qout
    qin.put(in_data)
    outdata=np.zeros((1,CHUNK),dtype=np.int16)
    if qout.qsize()>0:
        outdata=qout.get()
    return (outdata, pyaudio.paContinue)

# 输入音频参数设置
CHUNK = 1024 * 8
FORMAT = pyaudio.paInt16
CHANNEL = 1
RATE = 44100
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNEL, rate=RATE,\
    input=True,output=True, frames_per_buffer=CHUNK, stream_callback=callback)

# 作图的设置
mpl.rcParams['toolbar'] = 'None'
fig=plt.figure()
ax = fig.add_subplot(211)
ax1= fig.add_subplot(212)
#plt.subplot(212)
#plt.title('STFT Magnitude')
#plt.subplots_adjust(left=0.001, top=0.999,right=0.999,bottom=0.001)
#ax1.set_xlabel('Time [sec]', fontsize=12)
#ax1.set_ylabel('Frequency [Hz]', fontsize=12)
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.tight_layout()
fig.canvas.mpl_connect('key_press_event', on_press)
plt.get_current_fig_manager().set_window_title('Wave')
x = np.arange(0,CHUNK)
line0, = ax.plot(x, np.random.rand(CHUNK), color='#C04851')
line1, = ax.plot(x, np.random.rand(CHUNK), color='#FF5454')
ax.legend(['before STFT', 'after STFT'])
#设置坐标轴视图限制
ax.set_xlim(0,CHUNK-1)
ax.set_ylim(-2**15,2**15)
#打开交互模式
plt.ion()
plt.show()
#绘制stft后的频谱图
def drawstft(fre,ts,zxx,ax):
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    ax.set_ylim([fre[1], fre[-1]])
    ax.set_yscale('log')
    img = ax.pcolormesh(ts, fre, np.abs(zxx))
    fig.colorbar(img,ax=ax)
    return img

# 程序主体
while END==False:
    if qin.empty==True:
        continue
    indata=qin.get()
    qout.put(indata)
    #while not qin.empty():
    #    qin.get()
    data0 = np.frombuffer(indata, dtype=np.int16)
    if FIRST_CYCLE == False:    #data1有效
        cos_sim = analyse.cos_sim(data0=data0,data1=data1)
        print(cos_sim)
    fre, ts, zxx = signal.stft(data0,RATE,nperseg=512)
    if FIRST_CYCLE == True:
        img = drawstft(fre,ts,zxx,ax1)
    else:
        #img.set_array(np.abs(zxx))
        ax1.pcolormesh(ts, fre, np.abs(zxx))
    line0.set_ydata(data0)
    _, stft_data = signal.istft(zxx, RATE)
    line1.set_ydata(stft_data)
    fig.canvas.draw()
    fig.canvas.flush_events()
    data1 = data0
    FIRST_CYCLE = False
    plt.pause(0.01)

stream.stop_stream()
stream.close()
p.terminate()
