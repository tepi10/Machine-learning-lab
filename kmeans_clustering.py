import numpy as np
import matplotlib.pyplot as plt

# Generate dataset (realistic clusters)
np.random.seed(42)

data = np.vstack((
    np.random.normal(loc=[25, 30], scale=5, size=(50, 2)),
    np.random.normal(loc=[45, 60], scale=5, size=(50, 2)),
    np.random.normal(loc=[35, 80], scale=5, size=(50, 2))
))

# K-Means Algorithm
def kmeans(X, k, max_iters=100):
    centroids = X[np.random.choice(len(X), k, replace=False)]

    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)

        new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(k)])

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return clusters, centroids

# Run model
k = 3
clusters, centroids = kmeans(data, k)

print("Centroids:\n", centroids)

# Plot
plt.scatter(data[:, 0], data[:, 1], c=clusters)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200)

plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid()

plt.show()