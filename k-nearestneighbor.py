import math

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    x = np.linspace(-5, 5, 1000)
    y = x * x

    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts}',
        'font.size': 16
    })

    points_x = np.random.uniform(-5, 5, 10)
    points_y = points_x * points_x

    plt.plot(x, y, label='$f(x)=x^2$')
    plt.scatter(points_x, points_y, label='training set')

    y_predict_a = np.zeros_like(x)
    for i in range(len(x)):
        x_cur = x[i]
        x_one = math.inf
        x_two = math.inf
        y_one = math.inf
        y_two = math.inf
        for j in range(len(points_x)):
            if np.abs(x_cur - points_x[j]) < x_one:
                x_two = x_one
                y_two = y_one
                x_one = np.abs(x_cur - points_x[j])
                y_one = points_y[j]
            elif np.abs(x_cur - points_x[j]) < x_two:
                x_two = np.abs(x_cur - points_x[j])
                y_two = points_y[j]
        y_predict_a[i] = (y_one + y_two) * 0.5


    plt.xlabel("$x$")
    plt.ylabel("$y$")

    plt.plot(x, y_predict_a, label='$\hat f(x)$, unweighted average')

    plt.legend()

    plt.savefig("two-nearestneighbor.pdf", bbox_inches="tight")

    plt.show()