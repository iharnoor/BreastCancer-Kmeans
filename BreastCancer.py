# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def prediction_accuracy(cm):
    correct = cm[0][0] + cm[1][1]
    total = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
    return correct / total * 100


# Importing the dataset 569x30 dimensions
dataset = pd.read_csv('breast_data.csv')
X = dataset.iloc[:, :].values
dataset_true = pd.read_csv('breast_truth.csv')
y_true = dataset_true.iloc[:, [0]].values
print(X)

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans

# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('The Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters=2, init='k-means++')
y_kmeans = kmeans.fit_predict(X)
print()

# checking the efficiency
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_kmeans)
accuracy = prediction_accuracy(cm)
print(accuracy)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of People')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

""""(d) Run your algorithm several times, starting with different centers. Do your results change depending on this? Explain.
Ans: Yes the if we start the algorithm at different centers then we get different answers because of the Bad initialization step.
Our initialization step should use k-Means++ to solve the problem.
"""
"""(e)"""
