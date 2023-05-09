'''
Noise Reduction:
type slow when noise reduction on, not more than 15wpm
listen to foreground and background an data folder(present if noise reduction on) to see if keystroke sounds are filtered as noise or not
the background shound not have keystroke sounds in it
'''
noise_reduction=True
number_of_noise_reduction_loops=3

'''
cosine_similarity_width:
its the width(in seconds) for which median is calculated
in noise reduction (look at line 20-23 in clean_data.py)
basically increasing it might decrease background noise(and also foreground if overdone) 
its each element represent the width used in each loop
'''
cosine_similarity_width=[5,7,10]

plot_spectrogram=False

keystroke_duration_milliseconds = 50

'''
if thinking of appending already present data change these names
to data*.npy and labels*.npy (in regex so for example data1.py,labels1.py) and the old data will not we overwritten
then collect and append the data using combine_all_data.py script
'''
name_of_data_file='data.npy'
name_of_labels_file='labels.npy'



if len(cosine_similarity_width)!=number_of_noise_reduction_loops:cosine_similarity_width=[5 for _ in range(number_of_noise_reduction_loops)]
