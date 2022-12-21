'''
Noise Reduction:
type slow when noise reduction on, not more than 15wpm
listen to foreground and background an data folder(present if noise reduction on) to see if keystroke sounds are filtered as noise or not
the background shound not have keystroke sounds in it
'''
noise_reduction=True

'''
cosine_similarity_width:
its the width(in seconds) for which median is calculated
in noise reduction (look at line 17-20 in clean labeled data)
basically increasing it might decrease background noise(and also foreground of overdone) 
'''
cosine_similarity_width=5

plot_spectrogram=False

keystroke_duration_milliseconds = 50

'''
if thinking of appending already present data change these names
to d1.npy and l1.npy and the old data will not we overwritten
then collect and append the data using append_data.npy script
'''
name_of_data_file='data.npy'
name_of_labels_file='labels.npy'

