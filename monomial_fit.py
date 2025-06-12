#!/usr/bin/env python3
"""
Compute degree-12 minimax polynomial for cos(y) on [0, π/4]
and print in C++ constexpr form.
"""

import numpy as np
from numpy.polynomial import Polynomial
import scipy.linalg as la
import math

def remez_general(f, degree, a, b, n_iter=15, grid_n=2001):
    m = degree + 2
    xs = 0.5*(a+b) + 0.5*(b-a)*np.cos(np.pi * np.arange(m) / (m-1))
    xs.sort()

    for _ in range(n_iter):
        V = np.vander(xs, degree+1, increasing=True)
        signs = (-1)**np.arange(m)
        A = np.hstack([V, signs[:, None]])
        b_vec = f(xs)
        sol, *_ = la.lstsq(A, b_vec)
        c = sol[:-1]
        E = sol[-1]

        grid = np.linspace(a, b, grid_n)
        P_vals = Polynomial(c)(grid)
        err = P_vals - f(grid)
        idx = np.argpartition(-np.abs(err), m)[:m]
        xs = np.sort(grid[np.sort(idx)])

    return c, abs(E)

if __name__ == "__main__":
    # defaults
    degree = 5
    a = -math.pi / 4
    b = math.pi / 4
    f = np.cos

    coeffs, max_err = remez_general(f, degree, a, b)

    print(f"// degree-{degree} minimax for cos(y) on [0, π/4], max abs error ≈ {max_err:.3e}")
    for i, ci in enumerate(coeffs, start=0):
        # skip the constant term if you only want c1...cN, or adjust start=1
        print(f"constexpr double c{i+1} = {ci:.20e};")
