import math

import matplotlib.pyplot as plt
import numpy as np

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

    plt.savefig("maximumlikelihood-real.pdf", bbox_inches="tight")

    plt.show()

    mu_values = np.linspace(-6, 6, 1000)
    likelihood = mu_values * 0

    for point in points_x:
        likelihood += np.log(
            (1 / np.sqrt(2 * math.pi * sigma_squared)) *
            np.exp(-1 * np.pow(point - mu_values, 2) / (2 * sigma_squared))
        )

    max_index = np.argmax(likelihood)
    max_mu = [mu_values[max_index]]
    max_likelihood = [likelihood[max_index]]

    plt.plot(mu_values, likelihood, label='$\log p_{model}(\mathbb X; \\theta)$')
    plt.scatter(max_mu, max_likelihood, label=f'maximum likelihood estimator ($\mu = {"{:.2f}".format(max_mu[0])}$)')
    plt.xlabel('$\mu$')
    plt.ylabel('$\log p_{model}(\mathbb X; \\theta)$')
    plt.legend()

    plt.savefig("maximumlikelihood-prediction.pdf", bbox_inches="tight")

    plt.show()
