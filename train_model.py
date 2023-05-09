import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Load the training data
X = np.load("data/data.npy")
y = np.load("data/labels.npy")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a classifier 
''' 
Options are:
svm.SVC(), GradientBoostingClassifier(), 
RandomForestClassifier(), KNeighborsClassifier(), 
LogisticRegression()'''
clf = RandomForestClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# save the model
joblib.dump(clf, 'model/model.pkl') 

# Use the trained classifier to make predictions on the test data
y_pred = clf.predict(X_test)
print('pred:',y_pred)
print('truth:',y_test)
print("Accuracy:", clf.score(X_test, y_test))
# Save the predictions to a file
# np.save("predictions.npy", y_pred)