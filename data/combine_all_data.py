import glob
import numpy as np

# Load the main data and label arrays
data = np.load('data.npy')
labels = np.load('labels.npy')

# Find all the data and label files
data_filenames = glob.glob('data*.npy')
label_filenames = glob.glob('labels*.npy')

# remove the main files from the list
data_filenames.pop(data_filenames.index('data.npy'))
label_filenames.pop(label_filenames.index('labels.npy'))

# Iterate through the data and label files, and concatenate the data onto the main arrays
for data_filename, label_filename in zip(data_filenames, label_filenames):
    data1 = np.load(data_filename)
    labels1 = np.load(label_filename)
    data = np.concatenate((data, data1), axis=0)
    labels = np.concatenate((labels, labels1), axis=0)

# Save the concatenated data and labels back to disk
np.save('data.npy', data)
np.save('labels.npy', labels)
