#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <array>
#include <numeric>
#include <algorithm>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/optional/xoptional_assembly.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/generators/xbuilder.hpp> // Includes xt::zeros
#include <xtensor-blas/xlinalg.hpp>

constexpr double pi = 3.14159265358979323846;

//==============================================================================
// ChebFunND: N dims → M outputs, all in xtensor_fixed
//==============================================================================
template <typename T, size_t N, size_t M, typename F, int... Degrees>
class ChebFunND {
  static_assert(sizeof...(Degrees) == N, "Degree pack must match dimensionality");

  template <int... Ds>
  static constexpr auto make_coeffs_shape() {
    return xt::xshape<(Ds + 1)..., M>();
  }

  using coeffs_t = xt::xtensor_fixed<T, decltype(make_coeffs_shape<Degrees...>())>;
  using vecN_t = xt::xtensor_fixed<T, xt::xshape<N>>;
  using vecM_t = xt::xtensor_fixed<T, xt::xshape<M>>;

  coeffs_t coeffs_;
  vecN_t a_, b_;
  std::array<size_t, N> shape_;

public:
  ChebFunND(const vecN_t &a, const vecN_t &b, F f)
    : a_(a), b_(b) {
    // 1) Record each dimension’s size (degree + 1)
    for (size_t d = 0; d < N; ++d) {
      shape_[d] = coeffs_.shape()[d];
    }

    // 2) Sample f at Chebyshev nodes into coeffs_
    std::array<size_t, N> idx{};
    sample<0>(f, idx);

    // 3) For each output component: DCT along each dimension
    for (size_t o = 0; o < M; ++o) {
      xt::xdynamic_slice_vector slice_selector(N + 1);
      for (size_t d = 0; d < N; ++d) {
        slice_selector[d] = xt::all();
      }
      slice_selector[N] = static_cast<long>(o);
      // Get a dynamic view of the current output component
      auto coeffs_slice = xt::dynamic_view(coeffs_, slice_selector);

      // Perform DCT along each dimension
      for (size_t axis = 0; axis < N; ++axis) {
        size_t m = coeffs_slice.shape()[axis]; // Number of coefficients along this axis

        // Iterate over all combinations of indices for dimensions other than 'axis'
        std::vector<size_t> other_dims_shape;
        for (size_t d = 0; d < N; ++d) {
          if (d != axis) {
            other_dims_shape.push_back(coeffs_slice.shape()[d]);
          }
        }

        size_t num_other_dims = other_dims_shape.size();
        size_t total_slices = 1;
        for (size_t s : other_dims_shape) {
          total_slices *= s;
        }

        std::vector<size_t> current_indices(num_other_dims, 0);

        for (size_t i = 0; i < total_slices; ++i) {
          // Calculate current_indices based on linear index i and other_dims_shape
          size_t temp_i = i;
          for (int d = num_other_dims - 1; d >= 0; --d) {
            current_indices[d] = temp_i % other_dims_shape[d];
            temp_i /= other_dims_shape[d];
          }

          // Build the slice selector for the 1D line along the current axis
          xt::xdynamic_slice_vector line_selector(N);
          size_t current_other_dim_idx = 0;
          for (size_t d = 0; d < N; ++d) {
            if (d == axis) {
              line_selector[d] = xt::all(); // Select the entire dimension for the current axis
            } else {
              // Select the fixed index for dimensions other than the current axis
              line_selector[d] = static_cast<long>(current_indices[current_other_dim_idx++]);
            }
          }

          // Get the 1D dynamic view (the "line") along the current axis
          auto line = xt::dynamic_view(coeffs_slice, line_selector);

          // Perform DCT on the line (in-place)
          xt::xtensor<T, 1> dct = xt::zeros<T>({m});
          for (size_t k = 0; k < m; ++k) {
            T sum{};
            for (size_t j = 0; j < m; ++j) {
              sum += line(j) * std::cos(pi * k * (j + T(0.5)) / m);
            }
            dct(k) = sum * (T(2) / m);
          }
          dct(0) *= T(0.5);

          // Write back the DCT coefficients to the original slice
          for (size_t j = 0; j < m; ++j) {
            line(j) = dct(j);
          }
        }
      }
    }
  }

  vecM_t operator()(const vecN_t &x) const {
    // Map x from [a, b] to [-1, 1]
    vecN_t xt = (T(2) * x - a_ - b_) / (b_ - a_);

    // Evaluate Chebyshev polynomials T_k(xt) for each dimension
    std::vector<xt::xtensor<T, 1>> Tvals(N); // Use vector of 1D xtensors
    for (size_t d = 0; d < N; ++d) {
      size_t m = shape_[d]; // Degree + 1
      Tvals[d] = xt::zeros<T>({m}); // Corrected: Use xt::zeros free function
      if (m > 0)
        Tvals[d](0) = T(1);
      if (m > 1)
        Tvals[d](1) = xt(d);
      for (size_t k = 2; k < m; ++k) {
        Tvals[d](k) = T(2) * xt(d) * Tvals[d](k - 1) - Tvals[d](k - 2);
      }
    }

    // Evaluate the interpolated function using the coefficients and Tvals
    vecM_t result = xt::zeros<T>({M});
    std::array<size_t, N> idx{}; // Indices for the coefficients
    eval<0>(idx, T(1), Tvals, result);
    return result;
  }

private:
  // Recursive helper to sample the function at Chebyshev nodes
  template <size_t D_idx>
  void sample(F &f, std::array<size_t, N> &idx) {
    for (size_t i = 0; i < shape_[D_idx]; ++i) {
      idx[D_idx] = i;
      if constexpr (D_idx + 1 < N) {
        sample<D_idx + 1>(f, idx);
      } else {
        // We are at the innermost dimension, calculate the point and sample f
        vecN_t pt;
        for (size_t d = 0; d < N; ++d) {
          T theta = pi * (idx[d] + T(0.5)) / shape_[d]; // Chebyshev node angle
          pt(d) = T(0.5) * (a_(d) + b_(d)) + T(0.5) * (b_(d) - a_(d)) * std::cos(theta); // Map from [-1, 1] to [a, b]
        }
        auto out = f(pt); // Sample the function
        // Store the sampled values in the coefficients tensor
        for (size_t o = 0; o < M; ++o) {
          std::array<size_t, N + 1> full;
          for (size_t d = 0; d < N; ++d)
            full[d] = idx[d];
          full[N] = o; // Index for the output component
          coeffs_[full] = out(o);
        }
      }
    }
  }

  // Recursive helper to evaluate the interpolated function
  template <size_t D_idx>
  void eval(std::array<size_t, N> &idx, T prod,
            const std::vector<xt::xtensor<T, 1>> &Tvals, // Updated type
            vecM_t &acc) const {
    if constexpr (D_idx == N) {
      // We have a full set of indices, add the term to the accumulator
      for (size_t o = 0; o < M; ++o) {
        std::array<size_t, N + 1> full;
        for (size_t d = 0; d < N; ++d)
          full[d] = idx[d];
        full[N] = o;
        acc(o) += coeffs_[full] * prod;
      }
    } else {
      // Recurse through the dimensions
      for (size_t k = 0; k < shape_[D_idx]; ++k) {
        idx[D_idx] = k; // Set index for current dimension
        // Multiply by the Chebyshev polynomial value for this dimension and index
        eval<D_idx + 1>(idx, prod * Tvals[D_idx](k), Tvals, acc); // Updated access
      }
    }
  }
};

//==============================================================================
// Test harness
//==============================================================================
template <size_t D, size_t M, typename F>
void test_interp(const xt::xtensor_fixed<double, xt::xshape<D>> &a,
                 const xt::xtensor_fixed<double, xt::xshape<D>> &b,
                 F fcn) {
  auto run_test = [&](auto &cheb) {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(0, 1);
    xt::xtensor_fixed<double, xt::xshape<D>> pt;
    std::array<double, M> max_rel{};
    max_rel.fill(0.0); // Initialize max_rel to zeros

    for (int i = 0; i < 10000; ++i) {
      for (size_t d = 0; d < D; ++d) {
        double t = dist(rng);
        pt(d) = a(d) + t * (b(d) - a(d)); // Sample a random point in the interval [a, b]
      }
      auto approx = cheb(pt); // Get the interpolated value
      double s = xt::sum(pt)(); // Calculate the sum of components for the exact function
      xt::xtensor_fixed<double, xt::xshape<M>> exact;
      for (size_t k = 0; k < M; ++k)
        exact(k) = (k + 1.0) * std::exp(s); // Calculate the exact value

      // Compare approximate and exact values and update max relative error
      for (size_t k = 0; k < M; ++k) {
        double e = exact(k), p = approx(k);
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
  xt::xtensor_fixed<double, xt::xshape<D>> a, b;
  for (size_t i = 0; i < D; ++i) {
    a(i) = -1.0;
    b(i) = 1.0;
  } // Define the interval [a, b]
  // Define the test function f(x_1, ..., x_D) = ((1)*exp(sum(x_i)), (2)*exp(sum(x_i)), ...)
  auto f = [](auto &&xs) {
    double s = xt::sum(xs)();
    xt::xtensor_fixed<double, xt::xshape<M>> out;
    for (size_t k = 0; k < M; ++k)
      out(k) = (k + 1.0) * std::exp(s);
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
