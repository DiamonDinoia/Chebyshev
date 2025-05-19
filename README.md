# Multivariate Chebyshev Interpolation

This document explains the process of constructing a multidimensional Chebyshev interpolant for a function:

\[
f : \mathbb{R}^N \to \mathbb{R}^M
\]

The function is approximated as:

\[
f(x) \approx \sum_{k_0=0}^{d_0} \cdots \sum_{k_{N-1}=0}^{d_{N-1}} C_{k_0, \ldots, k_{N-1}, o} \cdot \prod_{d=0}^{N-1} T_{k_d}(x_d^*)
\]

## Step-by-step Breakdown

### 1. Change of Variables

Each coordinate \( x_d \in [a_d, b_d] \) is mapped to \( x_d^* \in [-1, 1] \):

\[
x_d^* = \frac{2x_d - (a_d + b_d)}{b_d - a_d}
\]

### 2. Chebyshev Nodes

The Chebyshev nodes on \([-1, 1]\) are:

\[
x_j^{(d)} = \cos\left(\frac{\pi (j + 0.5)}{n_d}\right), \quad j = 0, \ldots, n_d - 1
\]

These are mapped to \([a_d, b_d]\) using:

\[
x_j = \frac{a_d + b_d}{2} + \frac{b_d - a_d}{2} x_j^{(d)}
\]

### 3. Sampling and DCT

The function is evaluated at Chebyshev nodes and the coefficients are computed via the Discrete Cosine Transform (Type II):

\[
C_k = \frac{2}{m} \sum_{j=0}^{m-1} f(x_j) \cos\left(\frac{\pi k (j + 0.5)}{m}\right), \quad k = 0, \ldots, m-1
\]

Note: \( C_0 \) is scaled by 0.5.

### 4. Polynomial Evaluation

Chebyshev polynomials are computed recursively:

\[
T_0(x) = 1, \quad T_1(x) = x, \quad T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)
\]

### 5. Interpolation

The interpolant is evaluated by summing:

\[
f(x) \approx \sum_{\vec{k}} C_{\vec{k}, o} \prod_{d=0}^{N-1} T_{k_d}(x_d^*)
\]
