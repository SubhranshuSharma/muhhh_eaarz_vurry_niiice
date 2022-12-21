import pyaudio
import wave

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
fr=[]

try:
    print('start typing!, press ctrl-c when done')
    while True:
        data=stream.read(1024)
        print(type(data))
        fr.append(data)
except KeyboardInterrupt:
    pass
stream.close()
p.terminate()
sou=wave.open('unlabeled_data.wav','wb')
sou.setnchannels(1)
sou.setsampwidth(p.get_sample_size(pyaudio.paInt16))
sou.setframerate(44100)
sou.writeframes(b"".join(fr))
sou.close()
