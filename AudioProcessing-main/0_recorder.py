import pyaudio
import wave

p = pyaudio.PyAudio()

FORMAT = pyaudio.paInt16
FS = 44100
CHANNELS = 1
CHUNK = 1024
RECORD_SECOND = 10

stream = p.open(format=FORMAT, channels=CHANNELS,\
          rate=FS, input=True, frames_per_buffer=CHUNK)


frames = []
num_times = int(RECORD_SECOND * FS / CHUNK)

for i in range(num_times):
    data = stream.read(CHUNK)
    frames.append(data)

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open('output.wav', 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(FS)
wf.writeframes(b''.join(frames))
wf.close()
