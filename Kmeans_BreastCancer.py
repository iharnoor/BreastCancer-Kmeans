import numpy as np
import pandas as pd
import math

# Load the Wisconsin Diagnostic Breast Cancer dataset (breast_data.csv). You should obtain a data
# matrix with D = 30 features and N = 569 samples. Run K-means clustering on this data.

dataset = pd.read_csv('breast_data.csv')
X = dataset.iloc[:, :].values

truth = pd.read_csv('breast_truth.csv')
Y = truth.iloc[:].values
malign = []
benign = []

centroid_man = pd.read_csv('centroids.csv')
centroids_temp = centroid_man.iloc[:, :].values
centroids_mat = np.transpose(centroids_temp)

newmatrix = [0, 0]
newmatrix[1] = centroids_mat[0]
newmatrix[0] = centroids_mat[1]
parsedCentroids = np.array(newmatrix)

# print(parsedCentroids)

for i in range(len(Y)):
    if Y[i] == 1:
        malign.append(X[i])
    else:
        benign.append(X[i])

malign = np.array(malign)
benign = np.array(benign)
malignCentroids = malign.mean(axis=0)
benignCentroids = benign.mean(axis=0)
matrix = [0, 0]
matrix[1] = malignCentroids
matrix[0] = benignCentroids
true_centroids = np.array(matrix)


def euclidian(a, b):
    sum = 0.0
    for i, j in zip(a, b):
        sum += (i - j) ** 2
    return math.sqrt(sum)


def compare_centriod(a, b):
    for i, j in zip(a, b):
        for k, l in zip(i, j):
            if (k - l > 0.1):
                return 0
            else:
                return 1


def prediction_accuracy(cm):
    correct = cm[0][0] + cm[1][1]
    total = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
    return correct / total * 100


""""(a) Implement a function that performs K-means clustering. You can get started with the following code:"""


def kmeans(k, epsilon=0, distance='euclidian'):
    # List to store past centroid
    previous_centriods = []

    # set the distance calculation type
    if distance == 'euclidian':
        dist_method = euclidian

    # Define k centroids
    # prototypes = (X[np.random.randint(0, len(X[:]), size=k)])  # """For random centers"""
    prototypes = centroids_mat #"""for e) part parsed Centers in the given file mu_init
    # prototypes = true_centroids #for f) part parsed Centers in the given file
    # To see past centroids

    previous_centriods.append(prototypes)
    # keep track of centroids for every iteration
    prototypes_old = np.zeros(prototypes.shape)
    # To store clusters
    belongs_to = np.zeros((len(X[:]), 1))
    n = []
    norm = 0
    for i, j in zip(prototypes, prototypes_old):
        n.append(dist_method(i, j))
        norm = np.argmin(n)
    iteration = 0
    while iteration < 1000:
        iteration += 1
        for i, j in zip(prototypes, prototypes_old):
            n.append(dist_method(i, j))
            norm = np.argmax(n)
        # For each instance in the dataset
        prototypes_old = prototypes
        for index_instance, instance in enumerate(X):
            # Define a distance
            dist_vec = np.zeros((k, 1))
            # For each centroid
            for index_prototype, prototype in enumerate(prototypes):
                # Compute the distance between x and centriod
                dist_vec[index_prototype] = dist_method(prototype, instance)
            # Find the smallest distance, assign that distance to a cluster
            belongs_to[index_instance, 0] = np.argmin(dist_vec)
        tmp_prototypes = np.zeros((k, len(X[1, :])))
        # For each cluster k
        for index in range(len(prototypes)):
            # Get all the points assigned to a cluster
            instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
            # Find the mean of those points, this is our new centroid
            prototype = np.mean(X[instances_close], axis=0)
            # Add your centroid to the current list
            tmp_prototypes[index, :] = prototype
            # If the centroids are same after iteration
            if (compare_centriod(tmp_prototypes, prototypes)):
                return belongs_to
        # Set the new list to the current list
        prototypes = tmp_prototypes
        # Add our calculated centroids to our history for comparison
        previous_centriods.append(tmp_prototypes)
    # Return calculated centroids, history of them all, and assignnments for which data set belons to
    return belongs_to  # Calculating Accuracy


""" b) Implementing Kmeans on the data"""
count = 0
val = kmeans(2)
total = len(Y)
sum = 0
for i, j in zip(Y, val):
    if i == j:
        count += 1
print(count / len(Y))
"""c) Accuracy = 82.04 with the above code"""

"""d) as the matrix is randomly generated, the answer keeps on changing because of the bad initialization step that is the
answer changes if we choose different centroids.csv. So, Kmeans++ initialization step can be used to improve it and get the best
results."""
"""e) accuracy = 16.54% for the centroids provided. check above for the parsed file and commented"""
""" f) What if you initialize with the true centers, obtained using the true clustering?
        Ans: Accuracy = 85.73
        check above for the parsed data
"""
"""g) Supervised Learning: Logistic regression with 25:75 data split Accuracy = 97%"""
"""   Supervised Learning: K-Nearest Neighbours with 25:75 data split Accuracy = 95%
    Can be even higher with Neural network.
"""
"""   UnSupervised Learning: Hierarchical Clustering Accuracy = 77%"""
