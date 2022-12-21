import numpy as np
import joblib

# Load the data
X = np.load("unlabeled_data.npy")
clf = joblib.load('model.pkl')

# Use the trained classifier to make predictions on the test data
y_pred = clf.predict(X)
print('pred:',y_pred)

# Save the predictions to a file
# np.save("predictions.npy", y_pred)