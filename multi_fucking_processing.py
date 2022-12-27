import pyaudio
import wave
import numpy as np
import joblib
import multiprocessing
import time,librosa
import collections
import time
import soundfile

# Create a queue to store the data chunks for prediction
data_queue = multiprocessing.Queue()

# Create a function to run predictions on the data chunks in the queue
def predict(data_queue,mic_process):
    me_too_poor=False
    number_of_noise_reduction_loops=3
    was_noise_reduction_on=True
    sr=44100
    keystroke_duration_milliseconds=50
    cosine_similarity_width=[5,7,10]
    # Load the trained classifier
    clf = joblib.load('model/model.pkl')
    recorded_data = collections.deque(maxlen=14 * 44100)
    elapsed_time = 0
    print('\npredictor ready')
    try:
        while True:
            if not data_queue.empty():
                data = data_queue.get()
                recorded_data.extend(data)
                elapsed_time += 1024/44100
                # print(elapsed_time)
                if elapsed_time >= 14.0:
                    if me_too_poor==True:mic_process.terminate()
                    y = np.array(recorded_data).astype(float)
                    y=y.reshape(-1,1)
                    # elapsed_time=0
                    # soundfile.write('out.wav', y, samplerate=44100, subtype='PCM_16')
                    y_inv_b=y
                    if was_noise_reduction_on:
                        for i in range(number_of_noise_reduction_loops):
                            S_full, phase = librosa.magphase(librosa.stft(y_inv_b))

                            S_filter = librosa.decompose.nn_filter(S_full,
                                                                    aggregate=np.median,
                                                                    metric='cosine',
                                                                    width=int(librosa.time_to_frames(cosine_similarity_width[i], sr=sr)))

                            S_filter = np.minimum(S_full, S_filter)

                            margin_i, margin_v = 2, 10
                            power = 2

                            mask_i = librosa.util.softmask(S_filter,
                                                            margin_i * (S_full - S_filter),
                                                            power=power)

                            mask_v = librosa.util.softmask(S_full - S_filter,
                                                            margin_v * S_filter,
                                                            power=power)

                            S_foreground = mask_v * S_full
                            if i==0:y=librosa.griffinlim(S_foreground)
                            else:y+=librosa.griffinlim(S_foreground)
                    if 14<=elapsed_time<14.1 or 0<=((elapsed_time-14)%2)<0.1:
                        if 14<=elapsed_time<14.1:elapsed_time=14.1
                        elif 0<=((elapsed_time-14)%2)<0.1:elapsed_time=int(elapsed_time)+0.1
                        y=y[int(len(y)-2*44100):]
                        # Detect onsets in the data and crop it to 50 milliseconds
                        onset_indices = librosa.onset.onset_detect(y=y, sr=sr, units='samples')
                        num_samples_to_crop = librosa.time_to_samples(keystroke_duration_milliseconds / 1000, sr=sr)
                        cropped_audio = []
                        for onset_index in onset_indices:
                            start_index = onset_index
                            end_index = onset_index + num_samples_to_crop
                            if len(y) > end_index:
                                cropped_audio.append(y[start_index:end_index])
                        if len(cropped_audio==0):print('no keystrokes detected')
                        for data_chunk in cropped_audio:
                            pred = clf.predict(data_chunk)
                            print('pred:', pred)
            else:
                # Sleep for a short period of time if the queue is empty
                time.sleep(0.01)
    except KeyboardInterrupt:pass


def mic(data_queue):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
    print('Start typing! Press Ctrl-C when done.')
    try:
        f_time=time.time()
        while True:
            # Record data from the audio stream
            data = stream.read(1024)
            # print(time.time()-f_time)
            data=np.frombuffer(data, dtype=np.int16).astype(float) / 32768.0
            data_queue.put(data)
    except KeyboardInterrupt:
        # Close the audio stream and terminate PyAudio
        stream.close()
        p.terminate()

mic_process = multiprocessing.Process(target=mic, args=(data_queue,))
prediction_process = multiprocessing.Process(target=predict, args=(data_queue,mic_process))
mic_process.start()
prediction_process.start()
