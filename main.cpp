#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <array>
#include <numeric>
#include <algorithm>
#include <tuple>    // For std::tuple and std::tuple_cat
#include <utility>  // For std::index_sequence and std::apply

// Include necessary nda headers
#include <nda/nda.hpp>
#include <nda/layout/idx_map.hpp> // Included based on user's selection

constexpr double pi = 3.14159265358979323846;

//==============================================================================
// ChebFunND: N dims → M outputs, all in nda::array
//==============================================================================
template <typename T, size_t N, size_t M, typename F, int... Degrees>
class ChebFunND {
  static_assert(sizeof...(Degrees) == N, "Degree pack must match dimensionality");

  using coeffs_t = nda::array<T, N + 1>; // nda::array for coefficients
  using vecN_t = nda::array<T, 1>; // nda::array for N-dim points
  using vecM_t = nda::array<T, 1>; // nda::array for M-dim outputs

  coeffs_t coeffs_;
  vecN_t a_, b_;
  std::array<long, N> shape_; // Use long for shape elements as per nda

  // Helper to apply function call with a tuple of arguments
  template <typename Array, typename... Args>
  static auto apply_call(Array &arr, std::tuple<Args...> &&t) {
    return std::apply([&](auto &&... args) {
      return arr(std::forward<decltype(args)>(args)...);
    }, std::forward<std::tuple<Args...>>(t));
  }

public:
  // Constructor
  ChebFunND(const vecN_t &a, const vecN_t &b, F f)
    : a_(a), b_(b) {

    // Determine the shape of the coefficients tensor from Degrees and M
    std::array<long, N + 1> coeffs_shape_array = {static_cast<long>(Degrees + 1)..., static_cast<long>(M)};
    coeffs_.resize(coeffs_shape_array); // Resize the coefficients array using the array

    // 1) Record each dimension’s size (degree + 1)
    for (size_t d = 0; d < N; ++d) {
      shape_[d] = coeffs_.shape()[d];
    }

    // 2) Sample f at Chebyshev nodes into coeffs_
    std::array<long, N> idx{}; // Use long for indices as per nda
    sample<0>(f, idx);

    // 3) Perform DCT for each output component along each dimension using nested loops and direct indexing
    std::array<long, N + 1> current_coeffs_idx{}; // Index for the coeffs_ array

    for (size_t o = 0; o < M; ++o) {
      // Loop over output components
      // Initialize indices for the current output component
      for (size_t d = 0; d < N; ++d)
        current_coeffs_idx[d] = 0;
      current_coeffs_idx[N] = static_cast<long>(o);

      // Perform DCT along each dimension
      for (size_t axis = 0; axis < N; ++axis) {
        long m = coeffs_.shape()[axis]; // Number of coefficients along this axis for this dimension

        // Iterate through all elements, performing DCT along the current axis
        std::array<long, N> loop_indices{}; // Indices for the current N dimensions
        std::fill(loop_indices.begin(), loop_indices.end(), 0);

        long total_elements_in_slice = 1;
        for (size_t d = 0; d < N; ++d) {
          if (d != axis)
            total_elements_in_slice *= coeffs_.shape()[d];
        }

        for (long i = 0; i < total_elements_in_slice; ++i) {
          // Construct the indices for the current 1D slice along 'axis'
          std::array<long, N + 1> line_start_idx{};
          for (size_t d = 0; d < N; ++d)
            line_start_idx[d] = loop_indices[d];
          line_start_idx[N] = static_cast<long>(o); // Fix the output dimension index

          // Create a temporary vector to hold the 1D slice data
          std::vector<T> line_data(m);

          // Extract the 1D slice data
          for (long j = 0; j < m; ++j) {
            std::array<long, N + 1> element_idx = line_start_idx;
            element_idx[axis] = j;
            line_data[j] = std::apply([&](auto &&... args) {
              return coeffs_(std::forward<decltype(args)>(args)...);
            }, element_idx);
          }

          // Perform 1D DCT (Type 2) on the line_data manually
          std::vector<T> dct_data(m, T(0));

          for (long k = 0; k < m; ++k) {
            T sum{};
            for (long j = 0; j < m; ++j) {
              sum += line_data[j] * std::cos(pi * k * (j + T(0.5)) / m);
            }
            dct_data[k] = sum * (T(2) / m);
          }
          dct_data[0] *= T(0.5);

          // Write back the DCT coefficients from dct_data to the coeffs_ array
          for (long k = 0; k < m; ++k) {
            std::array<long, N + 1> element_idx = line_start_idx;
            element_idx[axis] = k;
            std::apply([&](auto &&... args) {
              coeffs_(std::forward<decltype(args)>(args)...) = dct_data[k];
            }, element_idx);
          }

          // Increment loop_indices, skipping the current axis
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

  // Evaluate the interpolated function
  vecM_t operator()(const vecN_t &x) const {
    // Map x from [a, b] to [-1, 1]
    vecN_t xt(N); // Resize xt
    for (size_t i = 0; i < N; ++i) {
      xt(static_cast<long>(i)) = (T(2) * x(static_cast<long>(i)) - a_(static_cast<long>(i)) - b_(static_cast<long>(i)))
                                 / (b_(static_cast<long>(i)) - a_(static_cast<long>(i)));
    }

    // Evaluate Chebyshev polynomials T_k(xt) for each dimension
    // Use a 2D nda::array to store Tvals, shape (N, max_degree + 1)
    long max_degree_plus_1 = 0;
    for (size_t s : shape_) {
      max_degree_plus_1 = std::max(max_degree_plus_1, static_cast<long>(s)); // Cast s to long for std::max
    }
    nda::array<T, 2> Tvals({static_cast<long>(N), max_degree_plus_1});
    // Tvals.fill(T(1)); // Replace fill with loop
    for (long i = 0; i < Tvals.shape()[0]; ++i) {
      for (long j = 0; j < Tvals.shape()[1]; ++j) {
        Tvals(i, j) = T(1);
      }
    }

    for (size_t d = 0; d < N; ++d) {
      if (shape_[d] > 1)
        Tvals(static_cast<long>(d), 1) = xt(static_cast<long>(d));
      for (long k = 2; k < shape_[d]; ++k)
        Tvals(static_cast<long>(d), k) = T(2) * xt(static_cast<long>(d)) * Tvals(static_cast<long>(d), k - 1) - Tvals(
                                             static_cast<long>(d), k - 2);
    }

    vecM_t result(M); // Resize result
    // result.fill(T(0)); // Replace fill with loop
    for (long i = 0; i < result.shape()[0]; ++i)
      result(i) = T(0);

    // Evaluate the interpolated function iteratively
    std::array<long, N> coeff_idx{}; // Indices for the coefficients tensor
    std::fill(coeff_idx.begin(), coeff_idx.end(), 0); // Initialize indices to 0

    long total_coeffs = 1;
    for (size_t s : shape_) {
      total_coeffs *= s;
    }

    // Declare nda_full_coeff_idx outside the inner loop
    // Use std::array<long, N+1> instead of typename coeffs_t::index_t
    std::array<long, N + 1> nda_full_coeff_idx_array;
    std::array<long, N + 1> full_coeff_idx; // Moved declaration outside the loop

    for (long i = 0; i < total_coeffs; ++i) {
      // Calculate the product of Chebyshev polynomial values for the current indices
      T current_prod = T(1);
      for (size_t d = 0; d < N; ++d) {
        current_prod *= Tvals(static_cast<long>(d), coeff_idx[d]);
      }

      // Add the term to the result for each output component
      for (size_t o = 0; o < M; ++o) {
        // std::array<long, N + 1> full_coeff_idx; // Declaration moved outside
        for (size_t d = 0; d < N; ++d)
          full_coeff_idx[d] = coeff_idx[d];
        full_coeff_idx[N] = static_cast<long>(o);

        // Copy indices to nda_full_coeff_idx_array
        for (size_t j = 0; j < N + 1; ++j)
          nda_full_coeff_idx_array[j] = full_coeff_idx[j];

        // Use std::apply to unpack the index array for the call operator
        auto coeff_value = std::apply([&](auto &&... args) {
          return coeffs_(std::forward<decltype(args)>(args)...);
        }, nda_full_coeff_idx_array);

        result(static_cast<long>(o)) += coeff_value * current_prod;
      }

      // Increment the indices (simulating nested loops)
      long current_dim = N - 1;
      while (current_dim >= 0) {
        // Loop until incrementing the first dimension
        coeff_idx[current_dim]++;
        if (coeff_idx[current_dim] < shape_[current_dim]) {
          break; // Move to the next set of indices
        } else {
          coeff_idx[current_dim] = 0; // Reset current dimension index
          if (current_dim == 0) {
            break; // All indices have been iterated through
          }
          current_dim--; // Move to the previous dimension
        }
      }
    }

    return result;
  }

private:
  // Recursive helper to sample the function at Chebyshev nodes
  template <size_t D_idx>
  void sample(F &f, std::array<long, N> &idx) {
    for (long i = 0; i < shape_[D_idx]; ++i) {
      idx[D_idx] = i;
      if constexpr (D_idx + 1 < N) {
        sample<D_idx + 1>(f, idx);
      } else {
        // We are at the innermost dimension, calculate the point and sample f
        vecN_t pt(N); // Resize pt
        for (size_t d = 0; d < N; ++d) {
          T theta = pi * (idx[d] + T(0.5)) / shape_[d]; // Chebyshev node angle
          pt(static_cast<long>(d)) = T(0.5) * (a_(static_cast<long>(d)) + b_(static_cast<long>(d))) + T(0.5) * (
                                       b_(static_cast<long>(d)) - a_(static_cast<long>(d))) * std::cos(theta);
          // Map from [-1, 1] to [a, b]
        }
        auto out = f(pt); // Sample the function
        // Store the sampled values in the coefficients tensor
        for (size_t o = 0; o < M; ++o) {
          std::array<long, N + 1> full;
          for (size_t d = 0; d < N; ++d)
            full[d] = idx[d];
          full[N] = static_cast<long>(o); // Index for the output component
          // Accessing coeffs_ using an array of indices
          // Use std::array<long, N+1> instead of typename coeffs_t::index_t
          std::array<long, N + 1> nda_full_coeff_idx_array; // nda index type
          for (size_t j = 0; j < N + 1; ++j)
            nda_full_coeff_idx_array[j] = full[j];

          // Use std::apply to unpack the index array for the call operator
          std::apply([&](auto &&... args) {
            coeffs_(std::forward<decltype(args)>(args)...) = out(static_cast<long>(o));
          }, nda_full_coeff_idx_array);
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
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(0, 1);
    nda::array<double, 1> pt(D); // Use nda::array for pt

    std::array<double, M> max_rel{};
    // max_rel.fill(0.0); // std::array fill should be fine, but using loop for consistency
    for (size_t i = 0; i < M; ++i)
      max_rel[i] = 0.0;

    for (int i = 0; i < 10000; ++i) {
      for (size_t d = 0; d < D; ++d) {
        double t = dist(rng);
        pt(static_cast<long>(d)) = a(static_cast<long>(d)) + t * (b(static_cast<long>(d)) - a(static_cast<long>(d)));
        // Sample a random point in the interval [a, b]
      }
      auto approx = cheb(pt); // Get the interpolated value
      double s = nda::sum(pt); // Use nda::sum
      nda::array<double, 1> exact(M); // Use nda::array for exact
      // exact.fill(0.0); // Replace fill with loop
      for (long k = 0; k < M; ++k)
        exact(k) = (k + 1.0) * std::exp(s); // Calculate the exact value

      // Compare approximate and exact values and update max relative error
      for (size_t k = 0; k < M; ++k) {
        double e = exact(static_cast<long>(k)), p = approx(static_cast<long>(k));
        double rel = std::abs(e) > 1e-12 ? std::abs(1 - p / e) : std::abs(p);
        // Calculate relative error, handle near-zero exact values
        max_rel[k] = std::max(max_rel[k], rel);
      }
    }
    std::cout << D << "-D -> " << M << " outputs, max rel errors:";
    for (double e : max_rel)
      std::cout << ' ' << e;
    std::cout << "\n";
  };

  // Instantiate ChebFunND with appropriate degrees based on dimensionality
  // Increased degrees slightly for potentially better accuracy with the same test function
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
  // Add more cases here if you need higher dimensions or different degrees
}

// Helper to run tests for different dimensions and output sizes
template <size_t D, size_t M>
void test_dim_out() {
  nda::array<double, 1> a(D), b(D); // Use nda::array for a and b
  for (size_t i = 0; i < D; ++i) {
    a(static_cast<long>(i)) = -1.0;
    b(static_cast<long>(i)) = 1.0;
  } // Define the interval [a, b]
  // Define the test function f(x_1, ..., x_D) = ((1)*exp(sum(x_i)), (2)*exp(sum(x_i)), ...)
  auto f = [](const nda::array<double, 1> &xs) {
    // Use nda::array for input
    double s = nda::sum(xs); // Use nda::sum
    nda::array<double, 1> out(M); // Use nda::array for output
    for (size_t k = 0; k < M; ++k)
      out(static_cast<long>(k)) = (k + 1.0) * std::exp(s);
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