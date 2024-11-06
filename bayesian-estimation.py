import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    # Pass 16 to the integer function for change of base
    return [int(hex_str[i:i + 2], 16) for i in range(1, 6, 2)]


def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1)) / 255
    c2_rgb = np.array(hex_to_RGB(c2)) / 255
    mix_pcts = [x / (n - 1) for x in range(n)]
    rgb_colors = [((1 - mix) * c1_rgb + (mix * c2_rgb)) for mix in mix_pcts]
    for i in range(len(rgb_colors)):
        rgb_colors[i] = [rgb_colors[i][0], rgb_colors[i][1], rgb_colors[i][2], 0.5]
    return rgb_colors


if __name__ == '__main__':
    sigma_squared = 1
    mu = -2

    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts}',
        'font.size': 16
    })

    mu_x = np.linspace(-5, 5, 1000)
    mu_y = np.zeros_like(mu_x)
    mu_y[(mu_x >= -4) & (mu_x <= 4)] = 1.0 / 8.0
    plt.plot(mu_x, mu_y, label="$p_{prior}$")
    plt.legend()
    plt.ylabel("$p$")
    plt.xlabel("$\\theta$")
    plt.savefig("bayesian-prior.pdf", bbox_inches="tight")
    plt.show()

    random_points = np.random.normal(mu, sigma_squared, 200)

    probability_points = np.zeros_like(random_points)

    for i in range(200):
        for j in range(100):
            theta = (8 / 99) * j + -4
            factor = 1.0 / 100
            for u in range(i):
                factor *= (1 / np.sqrt(2 * math.pi * sigma_squared)) * np.exp(
                    -1 * np.pow(random_points[u] - theta, 2) / (2 * sigma_squared))
                factor /= probability_points[u]
            probability_points[i] += (1 / np.sqrt(2 * math.pi * sigma_squared)) * np.exp(
                -1 * np.pow(random_points[i] - theta, 2) / (2 * sigma_squared)) * factor

    steps = [x for x in range(201)]

    x = np.linspace(-5, 5, 100)
    y_original = (1 / np.sqrt(2 * math.pi * sigma_squared)) * np.exp(-1 * np.pow(x - mu, 2) / (2 * sigma_squared))

    gradient = get_color_gradient('#eb3434', '#5eeb34', len(steps))

    for index in range(len(steps)):
        step = steps[index]
        y = np.zeros_like(x)
        for i in range(100):
            theta = (8 / 99) * i + -4
            p = (1 / np.sqrt(2 * math.pi * sigma_squared)) * np.exp(-1 * np.pow(x - theta, 2) / (2 * sigma_squared))
            p *= 1.0 / 100.0
            for u in range(step):
                p *= (1 / np.sqrt(2 * math.pi * sigma_squared)) * np.exp(
                    -1 * np.pow(random_points[u] - theta, 2) / (2 * sigma_squared))
                p /= probability_points[u]
            y += p
        print(f"{index} done")
        plt.plot(x, y, label=f"{step}", color=gradient[index])

    plt.plot(x, y_original, label="original curve, $\mathcal N(-2, 0)$", color='tab:blue')

    gradient = get_color_gradient('#eb3434', '#5eeb34', 3)
    plt.legend([
        Line2D([0], [0], color='tab:blue'),
        Line2D([0], [0], color=gradient[0]),
        Line2D([0], [0], color=gradient[1]),
        Line2D([0], [0], color=gradient[2])
    ], [
        'original curve, $\mathcal N(-2, 0)$',
        '$m=0$ data points',
        '$m=100$ data points',
        '$m=200$ data points'
    ])
    plt.xlabel('$x$')
    plt.ylabel('$p(x\;|\;\mathbb X)$')

    plt.savefig("bayesian-approx.pdf", bbox_inches="tight")

    plt.show()
