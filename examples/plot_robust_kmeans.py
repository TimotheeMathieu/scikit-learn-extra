# -*- coding: utf-8 -*-
"""
===================================================
A demo of using KMeans with RobustWeightedKMeans
===================================================
In this example we compare RobustWeightedKMeans to vanilla KMeans.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle

from sklearn_extra.robust import RobustWeightedKMeans

rng = np.random.RandomState(42)

# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)
X, labels_true = make_blobs(
    n_samples=300, centers=centers, cluster_std=0.3, random_state=rng
)

# Change the first 3 entries to outliers
for f in range(3):
    X[f] = [20, 3] + rng.normal(size=2) * 0.1

# Shuffle the data so that we don't know where the outlier is.
X = shuffle(X, random_state=rng)

# Compute the clusters with KMeans
kmeans = KMeans(n_clusters=3)
y = kmeans.fit_predict(X)

kmeans_rob = RobustWeightedKMeans(
    n_clusters=3,
    weighting="huber",
    max_iter=200,
    c=0.1,  # Measure the robustness of our estimation.
    random_state=rng,
)
kmeans_rob.fit(X)
yrob = kmeans_rob.predict(X)

fig, (axe1, axe2) = plt.subplots(1, 2)

axe1.scatter(X[:, 0], X[:, 1], c=y)
axe1.set_title("KMeans")
axe2.scatter(X[:, 0], X[:, 1], c=yrob)
axe2.set_title("RobustWeightedEstimator KMeans")
plt.show()
