import pyaudio,wave,os
import asyncio,time
import numpy as np
from pynput import keyboard
os.system('rm key_log.txt')
os.system('rm data/times.npy')
fr=[]
start_times=[]
times=[]
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
def on_press(key):
    global times
    try:
        times.append([time.time()-times[0][0],str(key)])
    except:
        times.append([time.time(),str(key)])
    with open('key_log.txt', 'a') as f:
        f.write(str(f"{key}".strip("'")))

listener = keyboard.Listener(on_press=on_press)

async def keylog():
    global times
    global fr
    try:
        listener.start()
        while True:
            await asyncio.sleep(0.01)  # yield control to other tasks
    except asyncio.CancelledError:
        pass
    finally:
        listener.stop()
    
async def mic():
    global start_time
    global fr
    print('start typing!, hit ctrl-c when done')
    try:
        start_times.append(time.time())
        while True:
            data=stream.read(1024)
            fr.append(data)
            await asyncio.sleep(0.01)  # yield control to other tasks
    except asyncio.CancelledError:
        pass
    finally:
        # close the audio stream and save the audio data to a file
        stream.close()
        p.terminate()
        sou=wave.open('data/raw_data.wav','wb')
        sou.setnchannels(1)
        sou.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        sou.setframerate(44100)
        sou.writeframes(b"".join(fr))
        sou.close()

async def main():
    tasks = [keylog(), mic()]
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except KeyboardInterrupt:
        for task in tasks:
            task.cancel()
            await asyncio.sleep(5)
        pass

try:
    asyncio.run(main())
except:pass
times=times[:[x[1] for x in times].index('Key.ctrl')]
times[0][0]=0
np.save('data/times.npy',times)
with open("key_log.txt", "r") as f:
    text = f.read()
    text,_ =text.split("Key.ctrl",1)
    text = text.replace("Key.space", "_")
    char_list = list(text)
    np.save("data/labels.npy", char_list)
    os.system('rm key_log.txt')
print(times)