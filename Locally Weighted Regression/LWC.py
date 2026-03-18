import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.linspace(-3, 3, 100)
y = np.sin(X) + np.random.normal(scale=0.2, size=X.shape)

def lwr_predict(x_query, X, y, tau=0.5):
    X_mat = np.vstack([np.ones(len(X)), X]).T
    y_mat = y.reshape(-1, 1)
    x_q = np.array([1, x_query]).reshape(1, -1)
    W = np.exp(-((X - x_query) ** 2) / (2 * tau ** 2))
    W_mat = np.diag(W)
    theta = np.linalg.pinv(X_mat.T @ W_mat @ X_mat) @ (X_mat.T @ W_mat @ y_mat)
    return (x_q @ theta).item()

X_test = np.linspace(-3, 3, 200)
y_pred = np.array([lwr_predict(xq, X, y) for xq in X_test])

plt.figure()
plt.scatter(X, y)
plt.plot(X_test, y_pred)
plt.title("Locally Weighted Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.show()

print("Experiment 9 Output: Graph displayed for Locally Weighted Regression")
