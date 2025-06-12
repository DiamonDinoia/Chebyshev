import numpy as np
import sys

# -----------------------------
# 1) Basic routines
# -----------------------------
# Define the target function
def f(x):
    return np.cos(x)

# Map from [-1, 1] to [a, b]
def map_to_domain(x, a, b):
    """
    Maps a value x from the interval [-1, 1] to a new interval [a, b].
    This is often used to transform Chebyshev nodes from their standard domain.
    """
    return 0.5 * (b - a) * x + 0.5 * (b + a)

# Generate Chebyshev nodes in [-1, 1]
def chebyshev_nodes(n):
    """
    Generates n Chebyshev nodes of the first kind in the interval [-1, 1].
    These nodes are optimal for polynomial interpolation because they minimize
    Runge's phenomenon.
    """
    k = np.arange(n)
    return np.cos((2 * k + 1) * np.pi / (2 * n))

# -----------------------------
# 2) LS‐based routines (General function fitting)
# -----------------------------

def ls_monomial_fit_residual_iterative(f, max_degree, x_train, iters=1):
    """
    Iteratively refines the Least Squares (LS) solution for a general function,
    fitting a polynomial with all powers of x up to `max_degree`.

    Args:
        f (function): The target function to approximate.
        max_degree (int): The highest power of x to include in the polynomial.
        x_train (np.array): The training points (x-coordinates) where the function is evaluated.
        iters (int): The number of residual-correction iterations to perform.
                     More iterations can improve accuracy by mitigating numerical errors.

    Returns:
        np.array: An array of monomial coefficients in descending order of power
                  (e.g., [c_n, c_{n-1}, ..., c_1, c_0] for a polynomial
                  c_n*x^n + ... + c_1*x + c_0), suitable for np.polyval.
    """
    # The number of coefficients needed for a polynomial of degree max_degree is max_degree + 1
    num_coeffs = max_degree + 1
    A = np.zeros((len(x_train), num_coeffs))

    # Construct the design matrix A: A_ij = x_train[i]**j
    # This matrix will contain columns for x^0, x^1, x^2, ..., x^max_degree
    for j in range(num_coeffs):
        A[:, j] = x_train**j # Powers are x^0, x^1, x^2, ..., x^max_degree

    y = f(x_train) # Evaluate the target function at the training points

    # 1) Initial LS solve: c0
    # The coefficients `c_all_powers` will be in ascending order of power (x^0, x^1, ...)
    c_all_powers, *_ = np.linalg.lstsq(A, y, rcond=None)

    # 2) Residual‐correction iterations
    # In each iteration, we calculate the residual (error) and solve for a correction
    # to the current coefficients. This helps reduce the error from floating-point arithmetic.
    for _ in range(iters):
        r     = y - A.dot(c_all_powers) # Calculate the residual (y_actual - y_predicted)
        δ, *_ = np.linalg.lstsq(A, r, rcond=None) # Solve for the correction delta
        c_all_powers = c_all_powers + δ # Add the correction to the coefficients

    # np.polyval expects coefficients in descending order of power (e.g., for x^n + ... + x^0).
    # Since `c_all_powers` are in ascending order, we reverse them before returning.
    coeffs_mono_descending = c_all_powers[::-1]

    return coeffs_mono_descending

# -----------------------------
# Function to find monomials for epsilon requirement (updated for general functions)
# -----------------------------
def get_monomials_for_epsilon(epsilon, f_target, a, b, max_degree=100, iters=10):
    """
    Finds the minimum polynomial degree and its monomial coefficients required
    to approximate a target function `f_target` on the interval [a, b]
    such that the maximum relative error is below a given `epsilon`.

    It uses Chebyshev nodes for training points and a residual-correction
    Least Squares fit.

    Args:
        epsilon (float): The target maximum relative error.
        f_target (function): The target function to approximate.
        a (float): The lower bound of the approximation interval.
        b (float): The upper bound of the approximation interval.
        max_degree (int): The maximum polynomial degree to attempt.
        iters (int): Number of residual-correction iterations for the LS fit.

    Returns:
        tuple: (monomial_coefficients, polynomial_degree, achieved_error)
               Returns (None, None, None) if the target epsilon cannot be met
               within the `max_degree`.
    """
    # Create a dense set of test points to evaluate the approximation error
    x_test = np.linspace(a, b, 1000)
    f_exact = f_target(x_test)

    # Iterate through polynomial degrees starting from 1 up to max_degree
    for n_poly in range(1, max_degree + 1):
        # Use 2*degree training points for the LS fit.
        # This provides enough data points for a robust fit.
        n_train = n_poly * 2

        # Generate Chebyshev nodes in [-1, 1] and map them to the domain [a, b]
        x_train_cheb = map_to_domain(chebyshev_nodes(n_train), a, b)

        # Use the general LS solver to get the monomial coefficients
        coeffs_mono_descending = ls_monomial_fit_residual_iterative(
            f_target, n_poly, x_train_cheb, iters=iters
        )

        # Evaluate the polynomial approximation at the test points
        f_approx = np.polyval(coeffs_mono_descending, x_test)

        # Calculate the maximum relative error.
        # np.finfo(float).eps is added to the denominator to prevent division by zero
        # if f_approx is very close to zero, ensuring numerical stability.
        # if f_exact and f_approx are both zero, the error is defined as zero.
        actual_error = np.max(np.abs(1 - f_exact / f_approx))

        # Check if the achieved error is within the target epsilon
        if actual_error <= epsilon:
            return coeffs_mono_descending, n_poly, actual_error

    # If the loop completes without meeting the epsilon, return None
    return None, None, None

# Example Usage: Using actual machine epsilon values
if __name__ == "__main__":
    a, b = 0, np.pi/4 # Define the approximation interval

    # Set NumPy print options for better readability of coefficients
    np.set_printoptions(precision=16, suppress=False, linewidth=120)

    # Get machine epsilon values for different float precisions
    machine_epsilon_float32 = np.finfo(np.float32).eps
    machine_epsilon_float64 = np.finfo(np.float64).eps

    print(f"Python's default float machine epsilon (double precision): {sys.float_info.epsilon:.2e}")
    print(f"NumPy's float32 machine epsilon (single precision):    {machine_epsilon_float32:.2e}")
    print(f"NumPy's float64 machine epsilon (double precision):   {machine_epsilon_float64:.2e}")

    # --- Test with single-like precision target epsilon ---
    target_epsilon_single_precision = machine_epsilon_float32
    print(f"\n--- Testing with target epsilon (single-like precision): {target_epsilon_single_precision:.2e} ---")
    monomial_coeffs_single, degree_single, error_single = get_monomials_for_epsilon(
        target_epsilon_single_precision, f, a, b, max_degree=100
    )

    if degree_single is not None:
        print(f"\nResults for target epsilon ({target_epsilon_single_precision:.0e}):")
        print(f"  Required polynomial degree: {degree_single}")
        print(f"  Achieved max relative error: {error_single:.2e}")
        print("  Monomial coefficients (descending order):")
        print(f"  {monomial_coeffs_single}")
    else:
        print(f"  Could not meet the target epsilon {target_epsilon_single_precision:.0e} within max degree.")

    # --- Test with double-like precision target epsilon ---
    # Using 4 times machine epsilon for a slightly looser but still high-precision target
    target_epsilon_double_precision = 4 * machine_epsilon_float64
    print(f"\n--- Testing with target epsilon (double-like precision): {target_epsilon_double_precision:.2e} ---")
    monomial_coeffs_double, degree_double, error_double = get_monomials_for_epsilon(
        target_epsilon_double_precision, f, a, b, max_degree=100
    )

    if degree_double is not None:
        print(f"\nResults for target epsilon ({target_epsilon_double_precision:.0e}):")
        print(f"  Required polynomial degree: {degree_double}")
        print(f"  Achieved max relative error: {error_double:.2e}")
        print("  Monomial coefficients (descending order):")
        print(f"  {monomial_coeffs_double}")
    else:
        print(f"  Could not meet the target epsilon {target_epsilon_double_precision:.0e} within max degree.")
