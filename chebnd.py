import numpy as np
import matplotlib.pyplot as plt

import itertools

def main():
    dim = 3

    if dim == 1:
        f = lambda x: np.exp(np.sin(4 * x[..., 0]))
    elif dim == 2:
        f = lambda x: (np.exp(np.sin(3 * x[..., 0]) * np.cos(2 * x[..., 1]))
                       + np.cos(x[..., 0] + x[..., 1]))
    elif dim == 3:
        f = lambda x: (np.exp(np.sin(3 * x[..., 0]) * np.cos(2 * x[..., 1]) *
            np.cos(x[..., 2]))
                       + np.cos(x[..., 0] + x[..., 1] - x[..., 2]))

    max_deg = 8
    n_samples = 10

    nodes_1d = [np.cos(np.pi * (i + 0.5) / n_samples) for i in range(n_samples)]

    nodes = dim * [nodes_1d]

    vals = f(np.array([x for x in itertools.product(*nodes)]))

    degs = dim * [range(max_deg)]

    vander = [[np.prod([_x ** _n for _x, _n in zip(x, n)])
               for n in itertools.product(*degs)]
              for x in itertools.product(*nodes)]

    vander = np.array(vander)

    coef, *_ = np.linalg.lstsq(vander, vals)

    est_vals = vander @ coef

    rel_err = np.linalg.norm(vals - est_vals) / np.linalg.norm(vals)

    print(f"Relative ℓ² error: {rel_err:e}")

    if dim == 1:
        plt.plot(nodes[0], vals)
        plt.plot(nodes[0], est_vals)
        plt.show()
    elif dim == 2:
        plt.subplot(1, 2, 1)
        plt.imshow(vals.reshape((n_samples, n_samples)))
        plt.subplot(1, 2, 2)
        plt.imshow(est_vals.reshape((n_samples, n_samples)))
        plt.show()


if __name__ == "__main__":
    main()
