import numpy as np
import matplotlib.pyplot as plt
import librosa, os
from settings import *
import soundfile as sf
import librosa.display

os.system("rm data/divided/*")

labels=np.load("data/labels.npy")
times=np.load('data/times.npy')
ti=[x[0] for x in times]
y, sr = librosa.load('data/raw_data.wav')
y_inv_b=y
os.system('rm data/background.wav data/foreground.wav')
if noise_reduction==True:
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
        S_background = mask_i * S_full
        if i==0:y=librosa.griffinlim(S_foreground)
        else:y+=librosa.griffinlim(S_foreground)
        y_inv_b=librosa.griffinlim(S_background)

sf.write('data/background.wav',y_inv_b,samplerate=sr,subtype='PCM_24')
sf.write('data/foreground.wav', y, samplerate=sr, subtype='PCM_24')

if plot_spectrogram==True:
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max),
                            y_axis='log', sr=sr)
    plt.title('Full spectrum')
    plt.colorbar()
    if noise_reduction==True:
        plt.subplot(3, 1, 2)
        librosa.display.specshow(librosa.amplitude_to_db(S_background, ref=np.max),
                                y_axis='log', sr=sr)
        plt.title('Background')
        plt.colorbar()
        plt.subplot(3, 1, 3)
        librosa.display.specshow(librosa.amplitude_to_db(S_foreground, ref=np.max),
                                y_axis='log', x_axis='time', sr=sr)
        plt.title('Foreground')
        plt.colorbar()
    plt.tight_layout()
    plt.show()
onset_indices = librosa.onset.onset_detect(y=y, sr=sr,units='samples')
onset_times = librosa.samples_to_time(onset_indices, sr=sr)
# onset_times = librosa.onset.onset_detect(y=y, sr=sr,units='time')
onset_times_shifted = [x - onset_times[0] for x in onset_times]
num_samples_to_crop = librosa.time_to_samples(keystroke_duration_milliseconds / 1000, sr=sr)
cropped_audio = []
for onset_index in onset_indices:
    start_index = onset_index
    end_index = onset_index + num_samples_to_crop
    if len(y)>end_index:
        cropped_audio.append(y[start_index:end_index])

ti = [float(x) for x in ti]
det=""
if len(onset_times_shifted)==len(ti):
    x=ti
    det=det+"perfect detections\n"
elif len(onset_times_shifted)>len(ti):
    x=ti
    det=det+'false detections, trying to correct\n'
elif len(onset_times_shifted)<len(ti):
    det=det+'false negatives, trying to correct\n'
    x=onset_times_shifted

# print(onset_times_shifted,times)

def remove_overdetections():
    global onset_times_shifted, cropped_audio,labels,ti,times
    # Sort both lists in increasing order
    onset_times_shifted.sort()
    ti.sort()

    # Initialize empty lists for the output
    cleaned_onset_times_shifted = []
    cleaned_ti = []
    cleaned_cropped_audio=[]
    cleaned_labels=[]
    cleaned_times=[]
    # Initialize indices for both lists
    i = 0
    j = 0
    while i < len(onset_times_shifted) and j < len(ti):
        # If the difference between the two values is within .1, append both values to the output lists
        # and increment both indices
        if abs(onset_times_shifted[i] - ti[j]) < .1:
            cleaned_onset_times_shifted.append(onset_times_shifted[i])
            cleaned_cropped_audio.append(cropped_audio[i])
            cleaned_ti.append(ti[j])
            cleaned_times.append(times[j])
            cleaned_labels.append(labels[j])
            i += 1
            j += 1
            # If the value in onset_times_shifted is greater than the value in ti, increment the index for ti
        elif onset_times_shifted[i] > ti[j]:
            j += 1
        # If the value in onset_times_shifted is lesser than the value in ti, increment the index for onset_times_shifted
        else:
            i += 1
    return cleaned_onset_times_shifted, cleaned_ti, cleaned_cropped_audio, cleaned_labels, cleaned_times

onset_times_shifted, ti, cropped_audio, labels, times = remove_overdetections()

try:
    for i in range(len(cropped_audio)):
        sf.write(f'data/divided/{i}.wav', cropped_audio[i], samplerate=sr, subtype='PCM_24')
except Exception as e:
    print(e)
    pass
# cropped_audio = [librosa.feature.mfcc(y=x, sr=sr) for x in cropped_audio]
cropped_audio=np.array(cropped_audio);print(cropped_audio.shape)
labels=np.array(labels)
np.save(f"data/{name_of_data_file}",cropped_audio)
if det=="perfect detections\n":print(det[:-1]);corr="detected"
elif cropped_audio.shape[0]==labels.shape[0]:print(det+'corrected!');corr="detected(and corrected)"
print(f"total {corr} keystrokes:",cropped_audio.shape[0])
print("total keystrokes(ground truth):",labels.shape[0],"\n")

os.system('rm data/labels.npy data/times.npy')
np.save(f"data/{name_of_labels_file}",labels)
np.save("data/times.npy",times)

print(f"keystroke {corr} times:","\n",onset_times_shifted,"\n")
print("keystroke times(ground truth):","\n",ti,"\n")
print("keys pressed:",labels)