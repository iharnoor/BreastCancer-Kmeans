# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def prediction_accuracy(cm):
    correct = cm[0][0] + cm[1][1]
    total = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
    return correct / total * 100


# Importing the dataset
dataset = pd.read_csv('breast_data.csv')
X = dataset.iloc[:, :].values

truth = pd.read_csv('breast_truth.csv')
y = truth.iloc[:].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(prediction_accuracy(cm))