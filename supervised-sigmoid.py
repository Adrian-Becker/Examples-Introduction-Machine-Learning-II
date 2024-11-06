import math

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts}',
        'font.size': 32
    })

    x = np.linspace(-10, 10, 1000)
    y = 1.0 / (1 + np.exp(-1 * x))

    plt.plot(x, y, label='$\sigma(x)={1 \over 1+e^{-x}}$')

    plt.legend(loc='upper left')
    plt.xlabel('$x$')

    plt.savefig("supervised-sigmoid.pdf", bbox_inches="tight")

    plt.show()
