import math

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    sigma_squared = 1
    mu = 0
    points_x = np.linspace(-10, 10, 1000)
    points_y = points_x * 2.5 + 1.5
    points_y_estimator = points_x * 2.9 + 0.5

    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts}'
    })

    plt.plot(points_x, points_y, label='original function')
    plt.plot(points_x, points_y_estimator, label='estimator')

    plt.xlabel("$x$")
    plt.ylabel("y")

    plt.legend()

    plt.savefig("estimator.pdf", bbox_inches="tight")

    plt.show()
