# Import necessary modules
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# Load the training data
X = np.load("data/data.npy")
y = np.load("data/labels.npy")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a dictionary of classifiers to be compared
classifiers = {
    "gradient boost": GradientBoostingClassifier(),
    "SVM": svm.SVC(),
    "Random Forest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "logistic regression": LogisticRegression()
}

# Train each classifier and store their performance
performance = {}
for name, clf in classifiers.items():
    scores = cross_val_score(clf, X, y, cv=5)
    performance[name] = scores.mean()

# Find the best classifier
best_classifier = max(performance, key=performance.get)
print("The best classifier is:", best_classifier)
print(performance)