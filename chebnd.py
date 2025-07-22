import itertools
import matplotlib.pyplot as plt
import numpy as np
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

    max_deg = 4
    n_samples = max_deg
    n_eval = 1000

    # Chebyshev-like nodes
    nodes_1d = [np.cos(np.pi * (i + 0.5) / n_samples) for i in range(n_samples)]
    # Build the full grid and sample
    nodes = [nodes_1d] * dim
    X_fit = np.array(list(itertools.product(*nodes)))
    y_fit = f(X_fit)

    # --- Method 1: Full least-squares solve ---
    start = time.perf_counter()
    degs = [range(max_deg)] * dim
    V_full = np.array([
        [np.prod([x[i] ** n[i] for i in range(dim)]) for n in itertools.product(*degs)]
        for x in X_fit
    ])
    coef_flat, *_ = np.linalg.lstsq(V_full, y_fit, rcond=None)
    coef_nd_full = coef_flat.reshape([max_deg] * dim)
    elapsed_full = time.perf_counter() - start
    print(f"Full least-squares solve time: {elapsed_full:.5f} sec")

    # --- Method 2: Separable 1D-by-1D solve ---
    start = time.perf_counter()
    nodes_arr = np.array(nodes_1d)
    V = np.vander(nodes_arr, N=max_deg, increasing=True)
    # sample tensor F of shape (n_samples,)*dim
    grid_pts = np.array(list(itertools.product(nodes_arr, repeat=dim)))
    F = f(grid_pts).reshape((n_samples,) * dim)

    # peel off one axis at a time
    coef_sep = F.copy()
    for axis in range(dim):
        # bring target axis to front
        coef_sep = np.moveaxis(coef_sep, axis, 0)   # shape: (n_samples, ...)
        flat = coef_sep.reshape((n_samples, -1))    # (n_samples, M)
        solved = np.linalg.solve(V, flat)           # (max_deg, M)
        # reshape & move axis back
        coef_sep = solved.reshape((max_deg,) + coef_sep.shape[1:])
        coef_sep = np.moveaxis(coef_sep, 0, axis)
    elapsed_sep = time.perf_counter() - start
    print(f"Separable 1D-by-1D solve time: {elapsed_sep:.5f} sec")

    # --- Method 3: Separable ND-by-1D solve (generic for any dim) ---
    start = time.perf_counter()
    # reuse V and F from above
    coef_nd = F.copy()
    for axis in range(dim):
        # move this axis to last
        coef_nd = np.moveaxis(coef_nd, axis, -1)   # shape (..., n_samples)
        pre_shape = coef_nd.shape[:-1]
        M = int(np.prod(pre_shape))
        flat_nd = coef_nd.reshape(M, n_samples)    # (M, n_samples)
        # solve along last dim for all M fibers
        solved_nd = np.linalg.solve(V, flat_nd.T).T # (M, max_deg)
        # reshape back & restore axis
        coef_nd = solved_nd.reshape(*pre_shape, max_deg)
        coef_nd = np.moveaxis(coef_nd, -1, axis)
    elapsed_nd = time.perf_counter() - start
    print(f"Separable ND-by-1D solve time: {elapsed_nd:.5f} sec")

    # --- Evaluation on random points ---
    np.random.seed(42)
    X_eval = np.random.uniform(-1, 1, size=(n_eval, dim))
    y_true = f(X_eval)

    y_est_full = np.array([horner_nd(coef_nd_full, x) for x in X_eval])
    err_full = np.linalg.norm(y_true - y_est_full) / np.linalg.norm(y_true)

    y_est_sep2 = np.array([horner_nd(coef_sep, x) for x in X_eval])
    err_sep2 = np.linalg.norm(y_true - y_est_sep2) / np.linalg.norm(y_true)

    y_est_sepND = np.array([horner_nd(coef_nd, x) for x in X_eval])
    err_sepND = np.linalg.norm(y_true - y_est_sepND) / np.linalg.norm(y_true)

    print(f"Rel. error [full]        : {err_full:e}")
    print(f"Rel. error [sep 1D-by-1D] : {err_sep2:e}")
    print(f"Rel. error [sep ND-by-1D] : {err_sepND:e}")

    # --- Plotting (as before) ---
    if dim == 1:
        idx = np.argsort(X_eval[:, 0])
        plt.plot(X_eval[idx, 0], y_true[idx], label='True')
        plt.plot(X_eval[idx, 0], y_est_full[idx], '--', label='Full LS')
        plt.plot(X_eval[idx, 0], y_est_sep2[idx], '-.', label='Sep 1D×1D')
        plt.plot(X_eval[idx, 0], y_est_sepND[idx], ':', label='Sep ND-by-1D')
        plt.legend()
        plt.title('1D Comparison')
        plt.show()
    else:
        plt.figure(figsize=(18, 5))

        # Full LS
        plt.subplot(1, 3, 1)
        plt.scatter(y_true, y_est_full, s=5)
        mn, mx = y_true.min(), y_true.max()
        plt.plot([mn, mx], [mn, mx], 'k--')
        plt.xlabel('True')
        plt.ylabel('Estimated')
        plt.title(f'Full LS (dim={dim})')

        # Separable 1D-by-1D
        plt.subplot(1, 3, 2)
        plt.scatter(y_true, y_est_sep2, s=5, color='C1')
        plt.plot([mn, mx], [mn, mx], 'k--')
        plt.xlabel('True')
        plt.ylabel('Estimated')
        plt.title('Sep 1D×1D')

        # Separable ND-by-1D
        plt.subplot(1, 3, 3)
        plt.scatter(y_true, y_est_sepND, s=5, color='C2')
        plt.plot([mn, mx], [mn, mx], 'k--')
        plt.xlabel('True')
        plt.ylabel('Estimated')
        plt.title('Sep ND-by-1D')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
