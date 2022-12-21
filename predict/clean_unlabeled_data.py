import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import librosa.display

cosine_similarity_width=5
keystroke_duration_milliseconds=50
was_noise_reduction_on=False
plot_spectrogram=False

y, sr = librosa.load('unlabeled_data.wav')
S_full, phase = librosa.magphase(librosa.stft(y))
if was_noise_reduction_on==True:
  S_filter = librosa.decompose.nn_filter(S_full,
                                        aggregate=np.median,
                                        metric='cosine',
                                        width=int(librosa.time_to_frames(cosine_similarity_width, sr=sr)))

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
  y_inv = librosa.griffinlim(S_foreground)
  y_inv_b=librosa.griffinlim(S_background)
if plot_spectrogram==True:
  plt.figure(figsize=(12, 8))
  plt.subplot(3, 1, 1)
  librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max),
                          y_axis='log', sr=sr)
  plt.title('Full spectrum')
  plt.colorbar()
  if was_noise_reduction_on==True:
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
if was_noise_reduction_on==False:
  y_inv=y
onset_indices = librosa.onset.onset_detect(y=y_inv, sr=sr,units='samples')
onset_times = librosa.onset.onset_detect(y=y_inv, sr=sr,units='time')
onset_times_shifted = [x - onset_times[0] for x in onset_times]
num_samples_to_crop = librosa.time_to_samples(keystroke_duration_milliseconds / 1000, sr=sr)
cropped_audio = []
for onset_index in onset_indices:
    start_index = onset_index
    end_index = onset_index + num_samples_to_crop
    if len(y_inv)>end_index:
        cropped_audio.append(y_inv[start_index:end_index])
np.save("unlabeled_data.npy",cropped_audio)