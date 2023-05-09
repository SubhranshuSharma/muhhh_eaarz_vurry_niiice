import numpy as np
X = np.load("unlabeled_data.npy")

use_tf_model=True

if use_tf_model==True:
    import librosa, tensorflow as tf
    clf = tf.keras.models.load_model ('../model/model.h5')
    X_mel = []
    for x in X:
        x_mel = librosa.feature.melspectrogram(y=x, sr=44100, n_fft=2048, hop_length=512, n_mels=128)
        X_mel.append(x_mel)
    X = np.array(X_mel)
else:
    import joblib
    clf = joblib.load('../model/model.pkl')

y_pred = clf.predict(X)
print('pred:',y_pred)

# Save the predictions to a file
# np.save("predictions.npy", y_pred)