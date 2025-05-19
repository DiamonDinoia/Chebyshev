#include <iostream>
#include <random>
#include <cmath>
#include <utility>
#include <vector>
#include <array>
#include <numeric>
#include <algorithm>
#include <tuple>    // For std::tuple and std::tuple_cat

// Include necessary nda headers
#include <nda/nda.hpp>

constexpr double pi = 3.14159265358979323846;

//==============================================================================
// ChebFunND: N dims → M outputs, all in nda::array
// Approximates a function f: R^N -> R^M using a multidimensional Chebyshev series.
//==============================================================================
template <typename T, size_t N, size_t M, typename F, int... Degrees>
class ChebFunND {
  static_assert(sizeof...(Degrees) == N, "Degree pack must match dimensionality");

  using coeffs_t = nda::array<T, N + 1>; // Coefficients tensor C[k_0, ..., k_{N-1}, o]
  using vecN_t = nda::array<T, 1>; // N-dim points x
  using vecM_t = nda::array<T, 1>; // M-dim outputs f(x)

  coeffs_t coeffs_;
  vecN_t a_, b_; // Lower and upper bounds of the N-dimensional box
  std::array<long, N> shape_; // Shape of the coefficient tensor in the first N dimensions (Degrees + 1)

public:
  // Constructor: Computes the Chebyshev coefficients of the function f.
  // The function f is sampled at the Chebyshev nodes, and a multidimensional
  // Discrete Cosine Transform (DCT) is performed to find the coefficients.
  ChebFunND(vecN_t a, vecN_t b, F f)
    : a_(std::move(a)), b_(std::move(b)) {

    // Determine the shape of the coefficients tensor: (Degree_0+1, ..., Degree_{N-1}+1, M)
    std::array<long, N + 1> coeffs_shape_array = {static_cast<long>(Degrees + 1)..., static_cast<long>(M)};
    coeffs_.resize(coeffs_shape_array);

    // 1) Record each dimension’s size (degree + 1)
    for (long d = 0; d < N; ++d) {
      shape_[d] = coeffs_.shape()[d];
    }

    // 2) Sample f at the N-dimensional Chebyshev nodes into coeffs_.
    // The Chebyshev nodes in 1D on [-1, 1] are cos(pi * (j + 0.5) / m) for j = 0, ..., m-1.
    // These are mapped to the interval [a, b] using x = 0.5*(a+b) + 0.5*(b-a)*cos(theta).
    std::array<long, N> idx{}; // Indices for the current node (j_0, ..., j_{N-1})
    sample<0>(f, idx);

    // 3) Perform the N-dimensional Discrete Cosine Transform (Type II)
    // This is done iteratively along each dimension for each output component.
    for (long o = 0; o < M; ++o) {
      // Loop over output components
      for (long axis = 0; axis < N; ++axis) {
        // Perform DCT along each dimension
        long m = coeffs_.shape()[axis]; // Number of nodes/coefficients along the current axis

        // Iterate through all 1D slices along 'axis' for the current output component 'o'.
        // This simulates nested loops over dimensions other than 'axis' and the output dimension.
        std::array<long, N> loop_indices{}; // Indices for dimensions other than 'axis'
        std::fill(loop_indices.begin(), loop_indices.end(), 0);

        long total_elements_in_slice = 1;
        for (long d = 0; d < N; ++d) {
          if (d != axis)
            total_elements_in_slice *= coeffs_.shape()[d];
        }

        for (long i = 0; i < total_elements_in_slice; ++i) {
          // Construct the base index for the current 1D slice along 'axis'.
          std::array<long, N + 1> line_start_idx{};
          for (long d = 0; d < N; ++d)
            line_start_idx[d] = loop_indices[d];
          line_start_idx[N] = o;

          // Extract the 1D slice data.
          std::vector<T> line_data(m);
          for (long j = 0; j < m; ++j) {
            std::array<long, N + 1> element_idx = line_start_idx;
            element_idx[axis] = j;
            line_data[j] = std::apply([&](auto &&... args) {
              return coeffs_(std::forward<decltype(args)>(args)...);
            }, element_idx);
          }

          // Perform 1D DCT (Type 2) on the line_data manually.
          // C_k = (2/m) * sum_{j=0}^{m-1} x_j * cos(pi * k * (j + 0.5) / m)
          // The k=0 term has a factor of 0.5.
          std::vector<T> dct_data(m, T(0));
          for (long k = 0; k < m; ++k) {
            T sum{};
            for (long j = 0; j < m; ++j) {
              sum += line_data[j] * std::cos(pi * k * (j + T(0.5)) / m);
            }
            dct_data[k] = sum * (T(2) / m);
          }
          dct_data[0] *= T(0.5); // Adjustment for the k=0 term

          // Write back the DCT coefficients from dct_data to the coeffs_ array.
          for (long k = 0; k < m; ++k) {
            std::array<long, N + 1> element_idx = line_start_idx;
            element_idx[axis] = k;
            std::apply([&](auto &&... args) {
              coeffs_(std::forward<decltype(args)>(args)...) = dct_data[k];
            }, element_idx);
          }

          // Increment loop_indices to move to the next slice.
          long current_dim = N - 1;
          while (current_dim >= 0) {
            if (current_dim == axis) {
              --current_dim;
              continue;
            }
            ++loop_indices[current_dim];
            if (loop_indices[current_dim] < coeffs_.shape()[current_dim]) {
              break;
            } else {
              loop_indices[current_dim] = 0;
              if (current_dim == 0) {
                break;
              }
              --current_dim;
            }
          }
        }
      }
    }
  }

  // Evaluate the interpolated function at a given point x.
  // The evaluation is done by summing the product of coefficients and Chebyshev polynomials.
  // P(x) = sum_{k_0=0}^{D_0} ... sum_{k_{N-1}=0}^{D_{N-1}} C[k_0, ..., k_{N-1}, o] * prod_{d=0}^{N-1} T_{k_d}(x_d)
  vecM_t operator()(const vecN_t &x) const {
    // Map x from the interval [a, b] in each dimension to [-1, 1].
    // xt_d = (2 * x_d - (a_d + b_d)) / (b_d - a_d)
    vecN_t xt(N);
    for (long i = 0; i < N; ++i) {
      xt(i) = (T(2) * x(i) - a_(i) - b_(i))
              / (b_(i) - a_(i));
    }

    // Evaluate Chebyshev polynomials T_k(xt) for each dimension and each degree up to the maximum degree.
    // T_0(x) = 1, T_1(x) = x, T_k(x) = 2x*T_{k-1}(x) - T_{k-2}(x) for k >= 2.
    nda::array<T, 2> Tvals({N, *std::max_element(shape_.begin(), shape_.end())});
    Tvals = T(1); // T_0(x) = 1

    for (long d = 0; d < N; ++d) {
      if (shape_[d] > 1)
        Tvals(d, 1) = xt(d);
      for (long k = 2; k < shape_[d]; ++k)
        Tvals(d, k) = T(2) * xt(d) * Tvals(d, k - 1) - Tvals(
                          d, k - 2);
    }

    vecM_t result(M); // Result vector for the M outputs
    result = T(0); // Initialize result to zero

    // Evaluate the interpolated function by summing over all coefficient indices.
    // This loop iterates through all combinations of indices (k_0, ..., k_{N-1}).
    std::array<long, N> coeff_idx{}; // Current coefficient indices (k_0, ..., k_{N-1})
    std::fill(coeff_idx.begin(), coeff_idx.end(), 0);

    long total_coeffs = 1;
    for (long s : shape_) {
      total_coeffs *= s;
    }

    std::array<long, N + 1> full_coeff_idx; // Full index including the output dimension

    for (long i = 0; i < total_coeffs; ++i) {
      // Calculate the product of Chebyshev polynomial values for the current indices (k_0, ..., k_{N-1}).
      // prod_{d=0}^{N-1} T_{k_d}(x_d)
      T current_prod = T(1);
      for (long d = 0; d < N; ++d) {
        current_prod *= Tvals(d, coeff_idx[d]);
      }

      // Add the term C[k_0, ..., k_{N-1}, o] * prod_{d=0}^{N-1} T_{k_d}(x_d) to the result for each output component 'o'.
      for (long o = 0; o < M; ++o) {
        for (long d = 0; d < N; ++d)
          full_coeff_idx[d] = coeff_idx[d];
        full_coeff_idx[N] = o;

        auto coeff_value = std::apply([&](auto &&... args) {
          return coeffs_(std::forward<decltype(args)>(args)...);
        }, full_coeff_idx);

        result(o) += coeff_value * current_prod;
      }

      // Increment the indices (k_0, ..., k_{N-1}) to move to the next term in the sum.
      // This simulates nested loops.
      long current_dim = N - 1;
      while (current_dim >= 0) {
        coeff_idx[current_dim]++;
        if (coeff_idx[current_dim] < shape_[current_dim]) {
          break; // Move to the next set of indices
        } else {
          coeff_idx[current_dim] = 0; // Reset current dimension index
          if (current_dim == 0) {
            break;
          }
          --current_dim; // Move to the previous dimension
        }
      }
    }

    return result;
  }

private:
  // Recursive helper function to sample the function f at the N-dimensional Chebyshev nodes.
  // The nodes are determined by the indices (j_0, ..., j_{N-1}) where j_d ranges from 0 to Degree_d.
  template <size_t D_idx>
  void sample(F &f, std::array<long, N> &idx) {
    for (long i = 0; i < shape_[D_idx]; ++i) {
      idx[D_idx] = i;
      if constexpr (D_idx + 1 < N) {
        sample<D_idx + 1>(f, idx);
      } else {
        // At the innermost dimension, construct the N-dimensional point corresponding to the current indices.
        vecN_t pt(N);
        for (long d = 0; d < N; ++d) {
          // Calculate the Chebyshev node in [-1, 1] for dimension d.
          T theta = pi * (idx[d] + T(0.5)) / shape_[d];
          // Map the node from [-1, 1] to the interval [a_d, b_d].
          pt(d) = T(0.5) * (a_(d) + b_(d)) + T(0.5) * (
                    b_(d) - a_(d)) * std::cos(theta);
        }
        auto out = f(pt); // Sample the function at the point.
        // Store the sampled values in the coefficients tensor at the corresponding index (j_0, ..., j_{N-1}, o).
        for (long o = 0; o < M; ++o) {
          std::array<long, N + 1> full;
          for (long d = 0; d < N; ++d)
            full[d] = idx[d];
          full[N] = o;

          std::apply([&](auto &&... args) {
            coeffs_(std::forward<decltype(args)>(args)...) = out(o);
          }, full);
        }
      }
    }
  }
};

//==============================================================================
// Test harness
//==============================================================================
template <size_t D, size_t M, typename F>
void test_interp(const nda::array<double, 1> &a,
                 const nda::array<double, 1> &b,
                 F fcn) {
  auto run_test = [&](auto &cheb) {
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0, 1);
    nda::array<double, 1> pt(D); // Random point in the N-dimensional box

    std::array<double, M> max_rel{};
    max_rel.fill(0.0);

    for (int i = 0; i < 10000; ++i) {
      // Sample a random point in the interval [a, b] for each dimension.
      for (long d = 0; d < D; ++d) {
        double t = dist(rng);
        pt(d) = a(d) + t * (b(d) - a(d));
      }
      auto approx = cheb(pt); // Interpolated value
      auto exact = fcn(pt); // Exact function value

      for (long k = 0; k < M; ++k) {
        double e = exact(k), p = approx(k);
        double rel = std::abs(e) > 1e-12 ? std::abs(1 - p / e) : std::abs(p);
        max_rel[k] = std::max(max_rel[k], rel);
      }
    }
    std::cout << D << "-D -> " << M << " outputs, max rel errors:";
    for (double e : max_rel)
      std::cout << ' ' << e;
    std::cout << "\n";
  };

  // Instantiate ChebFunND with appropriate degrees based on dimensionality.
  if constexpr (D == 1) {
    ChebFunND<double, 1, M, F, 8> cheb(a, b, fcn);
    run_test(cheb);
  }
  if constexpr (D == 2) {
    ChebFunND<double, 2, M, F, 8, 8> cheb(a, b, fcn);
    run_test(cheb);
  }
  if constexpr (D == 3) {
    ChebFunND<double, 3, M, F, 8, 8, 8> cheb(a, b, fcn);
    run_test(cheb);
  }
  if constexpr (D == 4) {
    ChebFunND<double, 4, M, F, 8, 8, 8, 8> cheb(a, b, fcn);
    run_test(cheb);
  }
}

// Helper to run tests for different dimensions and output sizes
template <size_t D, size_t M>
void test_dim_out() {
  nda::array<double, 1> a(D), b(D); // Interval [a, b]
  for (long i = 0; i < D; ++i) {
    a(i) = -1.0;
    b(i) = 1.0;
  }
  // Define the test function f(x_1, ..., x_D) = ((1)*exp(sum(x_i)), (2)*exp(sum(x_i)), ...)
  auto f = [](const nda::array<double, 1> &xs) {
    double s = nda::sum(xs);
    nda::array<double, 1> out(M);
    for (long k = 0; k < M; ++k)
      out(k) = (double(k) + 1.0) * std::exp(2.0 * s);
    return out;
  };
  test_interp<D, M>(a, b, f); // Run the interpolation test
}

int main() {
  // Run tests for various dimensions and output sizes
  test_dim_out<1, 1>();
  test_dim_out<2, 1>();
  test_dim_out<3, 1>();
  test_dim_out<2, 3>();
  test_dim_out<3, 2>();
  test_dim_out<4, 4>();
  return 0;
}