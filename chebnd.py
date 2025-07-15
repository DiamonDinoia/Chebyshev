import numpy as np
import matplotlib.pyplot as plt
import itertools


def horner_nd(coef_nd, x):
    """
    Evaluate a multivariate polynomial at point x using Horner's method.
    coef_nd should be an array of shape (d0, d1, ..., d_{dim-1}), where di is degree count for x[i].
    x is a 1D array of length dim.
    """
    dim = coef_nd.ndim

    def rec(c, axis):
        if axis == dim - 1:
            # Last dimension: 1D Horner
            res = 0.0
            for coeff in c[::-1]:
                res = res * x[axis] + coeff
            return res
        else:
            res = 0.0
            for sub_c in c[::-1]:
                res = res * x[axis] + rec(sub_c, axis + 1)
            return res

    return rec(coef_nd, 0)


def main():
    # Dimension: choose 1, 2, or 3
    dim = 2

    # Define target function f
    if dim == 1:
        f = lambda x: np.exp(np.sin(4 * x[..., 0]))
    elif dim == 2:
        f = lambda x: (
                np.exp(np.sin(3 * x[..., 0]) * np.cos(2 * x[..., 1]))
                + np.cos(x[..., 0] + x[..., 1])
        )
    elif dim == 3:
        f = lambda x: (
                np.exp(np.sin(3 * x[..., 0]) * np.cos(2 * x[..., 1]) * np.cos(x[..., 2]))
                + np.cos(x[..., 0] + x[..., 1] - x[..., 2])
        )

    max_deg = 16  # Degree count per dimension
    n_samples = 2 * max_deg  # Number of sample points per dimension for fitting
    n_eval = 1000  # Number of random evaluation points

    # Generate Chebyshev nodes for fitting
    nodes_1d = [np.cos(np.pi * (i + 0.5) / n_samples) for i in range(n_samples)]
    nodes = [nodes_1d] * dim

    # Evaluate function on fitting grid
    X_fit = np.array(list(itertools.product(*nodes)))  # Shape: (n_samples**dim, dim)
    y_fit = f(X_fit)

    # Build Vandermonde for least-squares fit
    degs = [range(max_deg)] * dim
    V = np.array([
        [np.prod([x[i] ** n[i] for i in range(dim)]) for n in itertools.product(*degs)]
        for x in X_fit
    ])

    # Solve for coefficients
    coef_flat, *_ = np.linalg.lstsq(V, y_fit, rcond=None)
    coef_nd = coef_flat.reshape([max_deg] * dim)

    # Generate random evaluation points in [-1, 1]^dim
    np.random.seed(0)
    X_eval = np.random.uniform(-1, 1, size=(n_eval, dim))
    y_true = f(X_eval)

    # Evaluate polynomial at random points using Horner's method
    y_est = np.array([horner_nd(coef_nd, x) for x in X_eval])

    # Compute relative L2 error
    rel_err = np.linalg.norm(y_true - y_est) / np.linalg.norm(y_true)
    print(f"Relative ℓ² error on random points: {rel_err:e}")

    # Plot comparison
    if dim == 1:
        # Sort for plotting
        idx = np.argsort(X_eval[:, 0])
        plt.plot(X_eval[idx, 0], y_true[idx], label='True')
        plt.plot(X_eval[idx, 0], y_est[idx], '--', label='Horner')
        plt.legend()
        plt.show()
    else:
        # Scatter true vs estimated
        plt.scatter(y_true, y_est, s=5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')
        plt.xlabel('True values')
        plt.ylabel('Estimated via Horner')
        plt.title(f'True vs Estimated (dim={dim})')
        plt.show()


if __name__ == "__main__":
    main()
