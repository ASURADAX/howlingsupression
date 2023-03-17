import pyroomacoustics as  pra
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
audio, sr = sf.read('MUSIC01.wav')

corner = np.array([[0, 0], [7, 0], [7, 5], [0, 5]]).T

room = pra.Room.from_corners(corner, fs=sr,
                              max_order=3,
                              materials=pra.Material(0.2, 0.15),
                              ray_tracing=True, air_absorption=True)
room.add_source([1, 1], signal=audio)
# add microphone
R = pra.circular_2D_array(center=[2.,2.], M=3, phi0=0, radius=0.3)
room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
# fig, ax = room.plot()
# ax.set_xlim([-1, 10])
# ax.set_ylim([-1, 10])
# fig.show()

room.image_source_model()
# fig, ax = room.plot(img_order=3)
# fig.set_size_inches(18.5, 10.5)
# fig.show()

#room.plot_rir()
#fig = plt.gcf()
#fig.set_size_inches(20, 10)
#plt.show()

room.simulate()
sf.write('modi_wav.wav', room.mic_array.signals.T, samplerate=sr)
audio1, sr = sf.read('modi_wav.wav')

wave_data0 = np.fromstring(audio, dtype=np.short)
wave_data0.shape = -1, 2
wave_data0 = wave_data0.T

wave_data1 = np.fromstring(audio1, dtype=np.short)
wave_data1.shape = -1, 2
wave_data1 = wave_data1.T
print(wave_data0.shape)
print(wave_data1.shape)
time = np.arange(0,wave_data0[0].shape[0])
# time = np.zeros(wave_data0[0].shape)
print(time)
print(wave_data0[0])
# 绘制波形
plt.subplot(211) 
plt.plot(time, wave_data0[0])
plt.subplot(212) 
plt.plot(time, wave_data1[0][0:wave_data0[0].shape[0]], c="g")
plt.xlabel("time (seconds)")
plt.show()
