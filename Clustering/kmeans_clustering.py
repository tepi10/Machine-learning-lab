import numpy as np
import matplotlib.pyplot as plt

# -------- Step 1: Get User Input --------
n = int(input("Enter number of data points: "))
data = []

print("Enter the data points (x y):")
for i in range(n):
    x, y = map(float, input(f"Point {i+1}: ").split())
    data.append([x, y])

X = np.array(data)

K = int(input("Enter number of clusters (K): "))

# -------- Step 2: Initialize Centroids --------
np.random.seed(0)
centroids = X[np.random.choice(n, K, replace=False)]

# -------- Step 3: Distance Function --------
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# -------- Step 4: K-Means Algorithm --------
for iteration in range(100):
    clusters = [[] for _ in range(K)]
    
    # Assign points to nearest centroid
    for point in X:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append(point)
    
    old_centroids = centroids.copy()
    
    # Update centroids
    for i in range(K):
        if len(clusters[i]) > 0:
            centroids[i] = np.mean(clusters[i], axis=0)
    
    # Stop if centroids don't change
    if np.allclose(old_centroids, centroids):
        break

# -------- Step 5: Display Results --------
print("\nFinal Centroids:")
print(centroids)

# -------- Step 6: Plot Clusters --------
colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']

plt.figure()
for i in range(K):
    cluster_points = np.array(clusters[i])
    if len(cluster_points) > 0:
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i % len(colors)], label=f"Cluster {i+1}")

plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200, label="Centroids")
plt.title("K-Means Clustering (User Input)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()
