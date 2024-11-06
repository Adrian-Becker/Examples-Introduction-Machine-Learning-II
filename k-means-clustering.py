import math
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

if __name__ == '__main__':
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts}',
        'font.size': 19
    })

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])

    mean = [0, 0]
    cov = [[1, 0], [0, 1]]

    points_x, points_y = np.random.multivariate_normal(mean, cov, 500).T
    points_x[000:200] -= 5.0
    points_x[200:400] += 5.0
    points_y[000:100] += 5.0
    points_y[200:300] += 5.0
    points_y[100:200] -= 5.0
    points_y[300:400] -= 5.0

    points_x_plot = np.copy(points_x)
    points_y_plot = np.copy(points_y)

    points_x = points_x.tolist()
    points_y = points_y.tolist()

    cluster = []
    for i in range(6):
        cluster.append([0., 0., 0., 0., 0])

    while len(points_x) > 0:
        i = random.randint(0, len(points_x) - 1)

        min_dist = math.inf
        best_cluster = None

        x = points_x[i]
        y = points_y[i]
        points_x.pop(i)
        points_y.pop(i)

        for j in range(len(cluster)):
            c = cluster[j]
            dist = (c[0] - x) ** 2 + (c[1] - y) ** 2
            if dist < min_dist:
                min_dist = dist
                best_cluster = j

        cluster[best_cluster][2] += x
        cluster[best_cluster][3] += y
        cluster[best_cluster][4] += 1
        cluster[best_cluster][0] = cluster[best_cluster][2] / cluster[best_cluster][4]
        cluster[best_cluster][1] = cluster[best_cluster][3] / cluster[best_cluster][4]

    colors = [
        (255 / 255, 183 / 255, 0 / 255, 0.5),
        (125 / 255, 255 / 255, 0 / 255, 0.5),
        (255 / 255, 0 / 255, 77 / 255, 0.5),
        (189 / 255, 0 / 255, 255 / 255, 0.5),
        (0 / 255, 135 / 255, 255 / 255, 0.5),
        (0 / 255, 255 / 255, 231 / 255, 0.5)
    ]

    inverse_colors = [
        '#0048ff',
        '#8400ff',
        '#00ffb3',
        '#44ff00',
        '#ff7700'
    ]

    acc = 5
    for i in range(20 * acc):
        x = i / acc - 10. + 1 / (2 * acc)
        for j in range(20 * 5):
            y = j / acc - 10. + 1 / (2 * acc)

            min_dist = math.inf
            best_cluster = None
            for u in range(len(cluster)):
                dist = (cluster[u][0] - x) ** 2 + (cluster[u][1] - y) ** 2
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = u

            ax.add_patch(
                matplotlib.patches.Rectangle((x - 1 / (2 * acc), y - 1 / (2 * acc)), 1 / acc, 1 / acc,
                                             color=colors[best_cluster]))
        print(i)

    ax.scatter(points_x_plot, points_y_plot)

    for i in range(len(cluster)):
        c = cluster[i]
        ax.scatter([c[0]], [c[1]], color='indigo')

    entries = [
        Line2D([0], [0], color='tab:blue', lw=4),
        Line2D([0], [0], color='indigo', lw=4),
    ]

    ax.legend(entries, ["training set", "cluster center"], loc='lower center')

    plt.xlabel("$x$")
    plt.ylabel("$y$")

    plt.savefig(f"{len(cluster)}-means-clustering.pdf", bbox_inches="tight")

    plt.show()
