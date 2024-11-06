import math

import matplotlib.pyplot as plt
import numpy as np


def estimate_likelihood_and_gradient(mu, points_x):
    likelihood = 0
    gradient = 0

    for point in points_x:
        likelihood -= np.log(
            (1 / np.sqrt(2 * math.pi * sigma_squared)) *
            np.exp(-1 * np.pow(point - mu, 2) / (2 * sigma_squared))
        )
        gradient -= (point - mu) / sigma_squared

    return likelihood, gradient


if __name__ == '__main__':
    sigma_squared = 1
    mu = 0
    points_x = np.random.normal(mu, sigma_squared, 10)
    points_y = (1 / np.sqrt(2 * math.pi * sigma_squared)) * np.exp(-1 * np.pow(points_x - mu, 2) / (2 * sigma_squared))

    x = np.linspace(-3, 3, 1000)
    y = (1 / np.sqrt(2 * math.pi * sigma_squared)) * np.exp(-1 * np.pow(x - mu, 2) / (2 * sigma_squared))

    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts}',
        'font.size': 16
    })

    plt.plot(x, y, label='density for $\mu=0$, $\sigma^2=0$')
    plt.scatter(points_x, points_y, label='$\mathbb{X}$')

    plt.xlabel("$x$")
    plt.ylabel("probability density")

    plt.legend(loc='lower center')

    plt.savefig("gradient-real.pdf", bbox_inches="tight")

    plt.show()

    mu_values = np.linspace(-6, 6, 1000)
    likelihood = mu_values * 0
    gradient = mu_values * 0

    for point in points_x:
        likelihood -= np.log(
            (1 / np.sqrt(2 * math.pi * sigma_squared)) *
            np.exp(-1 * np.pow(point - mu_values, 2) / (2 * sigma_squared))
        )
        gradient -= (point - mu_values) / sigma_squared

    sequence_mu = []
    sequence_likelihood = []
    best_mu = 0
    c_mu = -5.5
    current_gradient = math.inf
    epsilon = 1e-1 * 0.6
    while np.abs(current_gradient) > 1e-2:
        best_mu = c_mu
        c_likelihood, c_gradient = estimate_likelihood_and_gradient(c_mu, points_x)
        sequence_mu.append(c_mu)
        sequence_likelihood.append(c_likelihood)
        c_mu -= c_gradient * epsilon
        current_gradient = c_gradient
        print(np.abs(current_gradient))

    max_index = np.argmin(likelihood)
    max_mu = [mu_values[max_index]]
    max_likelihood = [likelihood[max_index]]

    plt.plot(mu_values, likelihood, label='$J(\\theta)=-\log p_{model}(\mathbb X; \\theta)$')
    plt.plot(mu_values, gradient, label='$\\nabla_\\theta J(\\theta)$')
    plt.plot(sequence_mu, sequence_likelihood, marker="o",
             label=f"estimates (final: $\mu={'{:.2f}'.format(best_mu)}$)")
    plt.xlabel('$\mu$')
    # plt.ylabel('$\log p_{model}(\mathbb X; \\theta)$')
    plt.legend()

    plt.savefig("gradient-prediction.pdf", bbox_inches="tight")

    plt.show()
