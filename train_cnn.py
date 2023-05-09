import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import librosa.display
import matplotlib.pyplot as plt

# Load the training data
X = np.load("data/data.npy")
y = np.load("data/labels.npy")
n_classes=len(set(y))
label_dict = {label: index for index, label in enumerate(set(y))}
y = [label_dict[label] for label in y]
y = to_categorical(y)
# Convert the raw audio data to mel spectrograms
X_mel = []
for x in X:
    x_mel = librosa.feature.melspectrogram(y=x, sr=44100, n_fft=2048, hop_length=512, n_mels=128)
    X_mel.append(x_mel)
X_mel = np.array(X_mel)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_mel, y, test_size=0.2, shuffle=True)


# Display the mel spectrogram for the first training example
# librosa.display.specshow(X_train[0], x_axis='time', y_axis='mel')
# plt.colorbar()
# plt.show()


with tf.device('/CPU:0'):
    # Define the CNN model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=X_train.shape[1:]),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# create a TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

# Train the model on the training data
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])

# Save the trained model
model.save('model/model.h5')

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
print('class names (note down for future reference):',label_dict)




