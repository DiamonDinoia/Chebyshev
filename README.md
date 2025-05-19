
# Multivariate Chebyshev Interpolation

This document explains the process of constructing a multidimensional Chebyshev interpolant for a function:

$$
f : \mathbb{R}^N \to \mathbb{R}^M
$$

We aim to approximate a multivariate function using Chebyshev polynomials:

$$
f(x) \approx \sum_{k_0=0}^{d_0} \cdots \sum_{k_{N-1}=0}^{d_{N-1}} C_{k_0, \ldots, k_{N-1}, o} \cdot \prod_{d=0}^{N-1} T_{k_d}(x_d^*)
$$

### Definitions

- $N$: Number of input dimensions (domain of $f$).
- $M$: Number of output dimensions (codomain of $f$).
- $x \in \mathbb{R}^N$: Input vector.
- $x_d \in [a_d, b_d]$: The $d$-th coordinate of the input.
- $x_d^* \in [-1, 1]$: Transformed coordinate mapped to Chebyshev domain.
- $T_k(x)$: Chebyshev polynomial of degree $k$.
- $C_{k_0, \ldots, k_{N-1}, o}$: Chebyshev coefficient for multi-index $(k_0, \ldots, k_{N-1})$ and output component $o$.
- $d_i$: Maximum degree in dimension $i$.

---

## Step-by-step Breakdown

### 1. Change of Variables

Each input component $x_d \in [a_d, b_d]$ is mapped to Chebyshev domain $x_d^* \in [-1, 1]$ via:

$$
x_d^* = \frac{2x_d - (a_d + b_d)}{b_d - a_d}
$$

### 2. Chebyshev Nodes

The Chebyshev nodes in 1D are:

$$
x_j^{(d)} = \cos\left(\frac{\pi (j + 0.5)}{n_d}\right), \quad j = 0, \ldots, n_d - 1
$$

These nodes are mapped back to the interval $[a_d, b_d]$ as:

$$
x_j = \frac{a_d + b_d}{2} + \frac{b_d - a_d}{2} \cdot x_j^{(d)}
$$

### 3. Discrete Cosine Transform (DCT)

We compute the Chebyshev coefficients via the Discrete Cosine Transform (Type II):

$$
C_k = \frac{2}{m} \sum_{j=0}^{m-1} f(x_j) \cos\left(\frac{\pi k (j + 0.5)}{m}\right), \quad k = 0, \ldots, m-1
$$

Note: $C_0$ is scaled by an additional factor of $0.5$.

### 4. Chebyshev Polynomial Evaluation

Chebyshev polynomials $T_k(x)$ are defined recursively:

$$
T_0(x) = 1, \quad T_1(x) = x, \quad T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)
$$

### 5. Final Interpolation

The interpolated value is computed by evaluating the full tensor product expansion:

$$
f(x) \approx \sum_{k_0 = 0}^{d_0} \cdots \sum_{k_{N-1} = 0}^{d_{N-1}} C_{k_0, \ldots, k_{N-1}, o} \cdot \prod_{d=0}^{N-1} T_{k_d}(x_d^*)
$$

---

## Diagram

Below is a diagram showing the full interpolation pipeline:

![Multivariate Chebyshev Interpolation](chebyshev_diagram.png)
