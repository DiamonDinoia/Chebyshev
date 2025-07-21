import itertools
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import time

def horner_nd(coef_nd, x):
    dim = coef_nd.ndim
    def rec(c, axis):
        if axis == dim - 1:
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

    max_deg = 16
    n_samples = 1 * max_deg
    n_eval = 1000

    nodes_1d = [np.cos(np.pi * (i + 0.5) / n_samples) for i in range(n_samples)]
    nodes = [nodes_1d] * dim
    X_fit = np.array(list(itertools.product(*nodes)))
    y_fit = f(X_fit)

    # --- Full solve timing ---
    start = time.perf_counter()

    degs = [range(max_deg)] * dim
    V = np.array([
        [np.prod([x[i] ** n[i] for i in range(dim)]) for n in itertools.product(*degs)]
        for x in X_fit
    ])
    coef_flat, *_ = np.linalg.lstsq(V, y_fit, rcond=None)
    coef_nd = coef_flat.reshape([max_deg] * dim)

    elapsed_full = time.perf_counter() - start
    print(f"Full least-squares solve time: {elapsed_full:.4f} sec")

    # --- Separable solve if dim == 2 ---
    if dim == 2:
        start = time.perf_counter()

        nodes_1d = np.array(nodes_1d)
        V = np.vander(nodes_1d, N=max_deg, increasing=True)
        A = np.empty((max_deg, n_samples))

        # Step 1: Solve in x direction (for fixed y_j)
        for j, yj in enumerate(nodes_1d):
            f_col = np.array([
                f(np.array([[xi, yj]]))[0] for xi in nodes_1d
            ])
            a_col = la.solve(V, f_col)
            A[:, j] = a_col

        # Step 2: Solve in y direction (for fixed x^i)
        coef_sep = np.empty((max_deg, max_deg))
        for i in range(max_deg):
            a_row = A[i, :]
            c_row = la.solve(V, a_row)
            coef_sep[i, :] = c_row

        elapsed_sep = time.perf_counter() - start
        print(f"Separable 1D-by-1D solve time: {elapsed_sep:.4f} sec")

    # --- Evaluation ---
    np.random.seed(42)
    X_eval = np.random.uniform(-1, 1, size=(n_eval, dim))
    y_true = f(X_eval)

    y_est = np.array([horner_nd(coef_nd, x) for x in X_eval])
    rel_err = np.linalg.norm(y_true - y_est) / np.linalg.norm(y_true)
    print(f"Relative ℓ² error on random points: {rel_err:e}")

    if dim == 2:
        y_est_sep = np.array([horner_nd(coef_sep, x) for x in X_eval])
        rel_err_sep = np.linalg.norm(y_true - y_est_sep) / np.linalg.norm(y_true)
        print(f"Relative ℓ² error (separable 1D solve): {rel_err_sep:e}")

    if dim == 1:
        idx = np.argsort(X_eval[:, 0])
        plt.plot(X_eval[idx, 0], y_true[idx], label='True')
        plt.plot(X_eval[idx, 0], y_est[idx], '--', label='Horner')
        plt.legend()
        plt.show()
    else:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_est, s=5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')
        plt.xlabel('True values')
        plt.ylabel('Estimated via Horner')
        plt.title(f'True vs Estimated (dim={dim})')

        if dim == 2:
            plt.subplot(1, 2, 2)
            plt.scatter(y_true, y_est_sep, s=5, color='orange')
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')
            plt.xlabel('True values')
            plt.ylabel('Separable fit')
            plt.title('True vs Separable 1D Fit')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
