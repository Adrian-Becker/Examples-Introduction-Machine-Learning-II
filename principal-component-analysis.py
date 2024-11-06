import math

import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

if __name__ == '__main__':
    mean = [0, 0]
    # cov = [[1, 0], [0, 0.01]]
    # rotated version (rotated by 30deg)
    cov = [
        [np.sqrt(3) / 2, -0.5],
        [1 / 200, np.sqrt(3) / 200]
    ]
    points = np.random.multivariate_normal(mean, cov, 100)
    points_x, points_y = points.T

    plt.scatter(points_x, points_y, label="uncompressed data")
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.show()

    points_transformed = np.matmul(np.transpose(points), points)

    eigenvalues, eigenvectors = LA.eig(points_transformed)
    index = np.argmax(eigenvalues)
    W = np.array([eigenvectors[index]])
    W /= np.linalg.norm(W)

    np.set_printoptions(threshold=sys.maxsize)

    points_compressed = np.zeros((100,))
    for i in range(100):
        points_compressed[i] = np.matmul(W, points[i])[0]

    # points_uncompressed = points_compressed * np.tile(W, (len(points_x), 1)).T
    points_uncompressed = np.zeros_like(points)
    for i in range(len(points_compressed)):
        points_uncompressed[i] = np.matmul(np.matmul(W.T, W), points[i])[0]

    points_uncompressed_x, points_uncompressed_y = points_uncompressed.T

    plt.scatter(points_compressed, np.zeros_like(points_compressed))
    plt.show()

    plt.scatter(points_uncompressed_x, points_uncompressed_y)
    plt.show()
