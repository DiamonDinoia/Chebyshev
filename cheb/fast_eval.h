#pragma once
#include <array>
#include <vector>
#include <cmath>
#include <complex>
#include <utility>
#include <cassert>
#include <functional>
#include <type_traits>
#include <numeric>
#include <limits>
#include <algorithm>
#include <iomanip>
#include <iostream>

namespace poly_eval {

// Removed the old `horner_step` to avoid confusion and potential type issues.

// NEW: Horner's unrolled recursion for P(x) = c[0] + c[1]*x + ... + c[N-1]*x^(N-1)
// Iterates conceptually forward from c[0] up to c[N-1] for the array indices.
// This implements P(x) = c[0] + x * (c[1] + x * (c[2] + ... x * c[N-1]))
template <std::size_t N_total, std::size_t current_idx, typename OutputType, typename InputType>
__always_inline static constexpr OutputType horner_forward_step(const std::array<OutputType, N_total> &c, InputType x) {
  if constexpr (current_idx == N_total - 1) {
    // Base case: the highest degree coefficient
    return c[current_idx];
  } else {
    // Recursive call needs to pass OutputType and InputType explicitly
    return std::fma(horner_forward_step<N_total, current_idx + 1, OutputType, InputType>(c, x), x, c[current_idx]);
  }
}


// -----------------------------------------------------------------------------
// function_traits: Helper to deduce input and output types from a callable
// -----------------------------------------------------------------------------
template <typename T> struct function_traits;

template <typename R, typename Arg>
struct function_traits<R(*)(Arg)> {
  using result_type = R;
  using arg0_type = Arg;
};

template <typename R, typename Arg>
struct function_traits<std::function<R(Arg)>> {
  using result_type = R;
  using arg0_type = Arg;
};

template <typename F, typename R, typename Arg>
struct function_traits<R(F::*)(Arg) const> {
  using result_type = R;
  using arg0_type = Arg;
};

template <typename F, typename R, typename Arg>
struct function_traits<R(F::*)(Arg)> {
  using result_type = R;
  using arg0_type = Arg;
};

template <typename T>
struct function_traits : function_traits<decltype(&T::operator())> {
};

// -----------------------------------------------------------------------------
// Buffer: Conditional type alias for std::vector or std::array
// -----------------------------------------------------------------------------
template <typename T, std::size_t N_compile_time_val>
using Buffer = std::conditional_t<
  N_compile_time_val == 0, std::vector<T>, std::array<T, N_compile_time_val>>;

// -----------------------------------------------------------------------------
// FuncEval: monomial least-squares fit using Chebyshev sampling
// (Runtime or Fixed-Size Compile-Time Storage, but fitting is runtime)
// -----------------------------------------------------------------------------
template <class Func, std::size_t N_compile_time = 0, std::size_t Iters_compile_time = 1>
class FuncEval {
public:
  using InputType = typename function_traits<Func>::arg0_type;
  using OutputType = typename function_traits<Func>::result_type;

  static constexpr std::size_t kDegreeCompileTime = N_compile_time;
  static constexpr std::size_t kItersCompileTime = Iters_compile_time;

  template <std::size_t CurrentN = N_compile_time,
            typename = std::enable_if_t<CurrentN == 0>>
  FuncEval(Func F, int n, InputType a, InputType b)
    : deg_(n), low(b - a), hi(b + a) {
    assert(deg_ > 0 && "Polynomial degree must be positive");
    coeffs_.resize(deg_);
    initialize_coeffs(F);
  }

  template <std::size_t CurrentN = N_compile_time,
            typename = std::enable_if_t<CurrentN != 0>>
  FuncEval(Func F, InputType a, InputType b)
    : deg_(static_cast<int>(CurrentN)), low(b - a), hi(b + a) {
    assert(deg_ > 0 && "Polynomial degree must be positive (template N > 0)");
    initialize_coeffs(F);
  }

  OutputType operator()(InputType pt) const noexcept {
    InputType xi = map_from_domain(pt);
    return horner(coeffs_, xi);
  }

  const Buffer<OutputType, N_compile_time> &coeffs() const noexcept {
    return coeffs_;
  }

private:
  int deg_;
  const InputType low, hi;
  Buffer<OutputType, N_compile_time> coeffs_;

  void initialize_coeffs(Func F) {
    std::vector<InputType> x_cheb_;
    std::vector<OutputType> y_cheb_;
    x_cheb_.resize(deg_);
    for (int k = 0; k < deg_; ++k) {
      x_cheb_[k] = static_cast<InputType>(std::cos((2.0 * k + 1.0) * M_PI / (2.0 * deg_)));
    }
    y_cheb_.resize(deg_);
    for (int i = 0; i < deg_; ++i) {
      y_cheb_[i] = F(map_to_domain(x_cheb_[i]));
    }
    std::vector<OutputType> newton = bjorck_pereyra(x_cheb_, y_cheb_);
    std::vector<OutputType> temp_monomial_coeffs = newton_to_monomial(newton, x_cheb_);
    assert(temp_monomial_coeffs.size() == coeffs_.size() && "Monomial coefficients size mismatch after conversion!");
    std::copy(temp_monomial_coeffs.begin(), temp_monomial_coeffs.end(), coeffs_.begin());
    refine_via_bjorck_pereyra(x_cheb_, y_cheb_);
  }

  template <class T> constexpr T map_to_domain(const T T_arg) const { return static_cast<T>(0.5 * (low * T_arg + hi)); }

  template <class T> constexpr T map_from_domain(const T T_arg) const {
    return static_cast<T>((2.0 * T_arg - hi) / low);
  }

  // Runtime Horner's: Processes coefficients from highest degree down to lowest.
  // This is the standard, numerically stable Horner's implementation.
  // The loop iterates from `c.size() - 1` (highest degree coefficient) down to `0` (constant term).
  static OutputType horner(const Buffer<OutputType, N_compile_time> &c, InputType x) noexcept {
    if (c.empty()) {
      return static_cast<OutputType>(0.0);
    }
    OutputType acc = c[c.size() - 1]; // Start with the highest degree coefficient
    for (int k = static_cast<int>(c.size()) - 2; k >= 0; --k) {
      acc = acc * x + c[k];
    }
    return acc;
  }

  std::vector<OutputType> bjorck_pereyra(const std::vector<InputType> &x,
                                         const std::vector<OutputType> &y) const {
    int n = deg_;
    std::vector<OutputType> a = y;
    for (int k = 0; k < n - 1; ++k) {
      for (int i = n - 1; i >= k + 1; --i) {
        a[i] = (a[i] - a[i - 1]) / static_cast<OutputType>(x[i] - x[i - k - 1]);
      }
    }
    return a;
  }

  static std::vector<OutputType> newton_to_monomial(const std::vector<OutputType> &alpha,
                                                    const std::vector<InputType> &nodes) {
    int n = static_cast<int>(alpha.size());
    std::vector<OutputType> c(1, static_cast<OutputType>(0.0));
    for (int i = n - 1; i >= 0; --i) {
      c.push_back(static_cast<OutputType>(0.0));
      for (int j = static_cast<int>(c.size()) - 1; j >= 1; --j) {
        c[j] = c[j - 1] - static_cast<OutputType>(nodes[i]) * c[j];
      }
      c[0] = -static_cast<OutputType>(nodes[i]) * c[0];
      c[0] += alpha[i];
    }
    if (static_cast<int>(c.size()) > n) {
      c.resize(n);
    }
    return c;
  }

  void refine_via_bjorck_pereyra(const std::vector<InputType> &x_cheb_,
                                 const std::vector<OutputType> &y_cheb_) {
    for (std::size_t pass = 0; pass < kItersCompileTime; ++pass) {
      std::vector<OutputType> r_cheb(deg_);
      for (int i = 0; i < deg_; ++i) {
        InputType xi = x_cheb_[i];
        OutputType p_val = horner(this->coeffs_, xi);
        r_cheb[i] = y_cheb_[i] - p_val;
      }
      std::vector<OutputType> newton_r = bjorck_pereyra(x_cheb_, r_cheb);
      std::vector<OutputType> mono_r = newton_to_monomial(newton_r, x_cheb_);
      assert(mono_r.size() == coeffs_.size() && "Refinement coefficients size mismatch!");
      for (int j = 0; j < deg_; ++j) {
        coeffs_[j] += mono_r[j];
      }
    }
  }
};

// -----------------------------------------------------------------------------
// Unified make_func_eval API (for runtime or fixed-size, runtime-fitted evaluation)
// -----------------------------------------------------------------------------

// Overload 1: For COMPILE-TIME degree N_compile_time (> 0)
template <std::size_t N_compile_time, std::size_t Iters_compile_time = 1, class Func>
auto make_func_eval(Func F,
                    typename function_traits<Func>::arg0_type a,
                    typename function_traits<Func>::arg0_type b) {
  return FuncEval<Func, N_compile_time, Iters_compile_time>(F, a, b);
}

// Overload 2: For RUNTIME degree 'n' (N_compile_time = 0 internally)
template <std::size_t Iters_compile_time = 1, class Func>
auto make_func_eval(Func F, int n,
                    typename function_traits<Func>::arg0_type a,
                    typename function_traits<Func>::arg0_type b) {
  return FuncEval<Func, 0, Iters_compile_time>(F, n, a, b);
}

// -----------------------------------------------------------------------------
// Helper to generate linearly spaced points (can be constexpr in C++20)
// -----------------------------------------------------------------------------
template <typename T, std::size_t N>
constexpr std::array<T, N> constexpr_linspace(T start, T end) {
  std::array<T, N> points{}; // Value-initialize to zero
  if (N == 0)
    return points; // Empty array

  if (N == 1) {
    points[0] = start;
    return points;
  }
  T step = (end - start) / static_cast<T>(N - 1);
  for (std::size_t i = 0; i < N; ++i) {
    points[i] = start + static_cast<T>(i) * step;
  }
  return points;
}

// Runtime version (for compatibility with std::vector based linspace in other APIs)
template <typename T>
std::vector<T> linspace(T start, T end, int num_points) {
  std::vector<T> points(num_points);
  if (num_points <= 1) {
    if (num_points == 1)
      points[0] = start;
    return points;
  }
  T step = (end - start) / static_cast<T>(num_points - 1);
  for (int i = 0; i < num_points; ++i) {
    points[i] = start + static_cast<T>(i) * step;
  }
  return points;
}

// -----------------------------------------------------------------------------
// make_func_eval that finds minimum N for a given error tolerance
// (C++20: eps, MaxN, NumEvalPoints as compile-time constants)
// This still uses the runtime FuncEval internally, so fitting happens at runtime.
// -----------------------------------------------------------------------------
#if __cplusplus >= 202002L
template <double eps_val, std::size_t MaxN_val, std::size_t NumEvalPoints_val, std::size_t Iters_compile_time = 1, class
          Func>
auto make_func_eval(Func F,
                    typename function_traits<Func>::arg0_type a,
                    typename function_traits<Func>::arg0_type b) {
  using InputType = typename function_traits<Func>::arg0_type;
  using OutputType = typename function_traits<Func>::result_type;

  static_assert(MaxN_val > 0, "Max polynomial degree must be positive.");
  static_assert(NumEvalPoints_val > 1, "Number of evaluation points must be greater than 1.");

  std::vector<InputType> eval_points = linspace(a, b, static_cast<int>(NumEvalPoints_val));

  for (int n = 1; n <= static_cast<int>(MaxN_val); ++n) {
    FuncEval<Func, 0, Iters_compile_time> current_evaluator(F, n, a, b);
    double max_observed_error = 0.0;

    for (const auto &pt : eval_points) {
      OutputType actual_val = F(pt);
      OutputType poly_val = current_evaluator(pt);
      double current_abs_error = std::abs(1.0 - std::abs(poly_val / actual_val));
      if (current_abs_error > max_observed_error) {
        max_observed_error = current_abs_error;
      }
    }

    if (max_observed_error <= eps_val) {
      std::cout << "Converged: Found min degree N = " << n
          << " (Max Error: " << std::scientific << std::setprecision(4) << max_observed_error
          << " <= Epsilon: " << eps_val << ")\n";
      return current_evaluator;
    }
  }

  std::cout << "Warning: Did not converge to epsilon " << std::scientific << std::setprecision(4) << eps_val
      << " within MaxN = " << MaxN_val << ". Returning FuncEval with degree " << MaxN_val << ".\n";
  return FuncEval<Func, 0, Iters_compile_time>(F, static_cast<int>(MaxN_val), a, b);
}
#endif // __cplusplus >= 202002L

// -----------------------------------------------------------------------------
// make_func_eval that finds minimum N for a given error tolerance
// (C++17 and earlier: eps as runtime, MaxN, NumEvalPoints as compile-time template arguments)
// -----------------------------------------------------------------------------
template <std::size_t MaxN_val, std::size_t NumEvalPoints_val, std::size_t Iters_compile_time = 1, class Func>
auto make_func_eval(Func F, double eps, // eps as a runtime parameter
                    typename function_traits<Func>::arg0_type a,
                    typename function_traits<Func>::arg0_type b) {
  using InputType = typename function_traits<Func>::arg0_type;
  using OutputType = typename function_traits<Func>::result_type;

  static_assert(MaxN_val > 0, "Max polynomial degree must be positive.");
  static_assert(NumEvalPoints_val > 1, "Number of evaluation points must be greater than 1.");

  // Validate eps: cannot be less than machine precision for the output type
  if (eps < std::numeric_limits<double>::epsilon()) {
    if constexpr (std::is_floating_point_v<OutputType>) {
      if (eps < std::numeric_limits<OutputType>::epsilon()) {
        std::cerr << "Warning: Requested epsilon " << eps
            << " is less than machine epsilon for OutputType ("
            << std::numeric_limits<OutputType>::epsilon() << "). Clamping.\n";
        eps = std::numeric_limits<OutputType>::epsilon();
      }
    } else if constexpr (std::is_same_v<OutputType, std::complex<float>>) {
      if (eps < std::numeric_limits<float>::epsilon()) {
        std::cerr << "Warning: Requested epsilon " << eps
            << " is less than machine epsilon for std::complex<float> ("
            << std::numeric_limits<float>::epsilon() << "). Clamping.\n";
        eps = std::numeric_limits<float>::epsilon();
      }
    } else if constexpr (std::is_same_v<OutputType, std::complex<double>>) {
      if (eps < std::numeric_limits<double>::epsilon()) {
        std::cerr << "Warning: Requested epsilon " << eps
            << " is less than machine epsilon for std::complex<double> ("
            << std::numeric_limits<double>::epsilon() << "). Clamping.\n";
        eps = std::numeric_limits<double>::epsilon();
      }
    }
  }

  std::vector<InputType> eval_points = linspace(a, b, static_cast<int>(NumEvalPoints_val));

  for (int n = 1; n <= static_cast<int>(MaxN_val); ++n) {
    FuncEval<Func, 0, Iters_compile_time> current_evaluator(F, n, a, b);
    double max_observed_error = 0.0;
    for (const auto &pt : eval_points) {
      OutputType actual_val = F(pt);
      OutputType poly_val = current_evaluator(pt);
      double current_abs_error = std::abs(1.0 - std::abs(poly_val / actual_val));
      if (current_abs_error > max_observed_error) {
        max_observed_error = current_abs_error;
      }
    }
    if (max_observed_error <= eps) {
      std::cout << "Converged: Found min degree N = " << n
          << " (Max Error: " << std::scientific << std::setprecision(4) << max_observed_error
          << " <= Epsilon: " << eps << ")\n";
      return current_evaluator;
    }
  }
  std::cout << "Warning: Did not converge to epsilon " << std::scientific << std::setprecision(4) << eps
      << " within MaxN = " << MaxN_val << ". Returning FuncEval with degree " << MaxN_val << ".\n";
  return FuncEval<Func, 0, Iters_compile_time>(F, static_cast<int>(MaxN_val), a, b);
}

// -----------------------------------------------------------------------------
// ConstexprFuncEval: A dedicated class for compile-time polynomial fitting.
// All operations are constexpr.
// -----------------------------------------------------------------------------
template <class Func, std::size_t N_DEGREE, std::size_t ITERS = 1>
class ConstexprFuncEval {
public:
  using InputType = typename function_traits<Func>::arg0_type;
  using OutputType = typename function_traits<Func>::result_type;

  static_assert(N_DEGREE > 0, "Polynomial degree must be positive for constexpr evaluation.");

  // Constructor that performs the fitting at compile-time
  constexpr ConstexprFuncEval(Func F, InputType a, InputType b)
    : low(b - a), hi(b + a), coeffs_(initialize_coeffs(F, a, b)) {
  }

  // Evaluate interpolant at compile-time or runtime
  constexpr OutputType operator()(InputType pt) const noexcept {
    InputType xi = map_from_domain(pt);
    return horner(coeffs_, xi);
  }

  // Access monomial coefficients (lowest→highest)
  constexpr const std::array<OutputType, N_DEGREE> &coeffs() const noexcept {
    return coeffs_;
  }

private:
  const InputType low, hi; // low = b-a, hi = b+a
  std::array<OutputType, N_DEGREE> coeffs_;

  // Private constexpr helper functions
  constexpr InputType map_to_domain(const InputType T_arg) const { return 0.5 * (low * T_arg + hi); }
  constexpr InputType map_from_domain(const InputType T_arg) const { return (2.0 * T_arg - hi) / low; }

  // constexpr horner evaluation for std::array using horner_forward_step
  static constexpr OutputType horner(const std::array<OutputType, N_DEGREE> &c, InputType x) noexcept {
    if constexpr (N_DEGREE == 0) {
      return static_cast<OutputType>(0.0);
    } else {
      // Call the recursive forward step starting from the first coefficient (index 0)
      return horner_forward_step<N_DEGREE, 0, OutputType, InputType>(c, x);
    }
  }

  // constexpr Björck–Pereyra Newton solver
  static constexpr std::array<OutputType, N_DEGREE>
  bjorck_pereyra_constexpr(const std::array<InputType, N_DEGREE> &x,
                           const std::array<OutputType, N_DEGREE> &y) noexcept {
    std::array<OutputType, N_DEGREE> a = y;
    for (std::size_t k = 0; k < N_DEGREE - 1; ++k) {
      for (std::size_t i = N_DEGREE - 1; i >= k + 1; --i) {
        assert(x[i] - x[i - k - 1] != static_cast<InputType>(0.0));
        a[i] = (a[i] - a[i - 1]) / static_cast<OutputType>(x[i] - x[i - k - 1]);
      }
    }
    return a;
  }

  // constexpr conversion from Newton to monomial basis
  static constexpr std::array<OutputType, N_DEGREE>
  newton_to_monomial_constexpr(const std::array<OutputType, N_DEGREE> &alpha,
                               const std::array<InputType, N_DEGREE> &nodes) noexcept {
    std::array<OutputType, N_DEGREE> c{};

    if (N_DEGREE > 0) {
      c[N_DEGREE - 1] = alpha[N_DEGREE - 1];
    }

    for (int i = static_cast<int>(N_DEGREE) - 2; i >= 0; --i) {
      for (int j = static_cast<int>(N_DEGREE) - 1; j > 0; --j) {
        c[j] = c[j - 1] - static_cast<OutputType>(nodes[i + 1]) * c[j];
      }
      c[0] = -static_cast<OutputType>(nodes[i + 1]) * c[0];
      if (i >= 0) {
        c[0] += alpha[i];
      }
    }
    return c;
  }

  // Helper to initialize coefficients at compile-time
  constexpr std::array<OutputType, N_DEGREE> initialize_coeffs(Func F, InputType a, InputType b) {
    std::array<InputType, N_DEGREE> x_cheb_nodes{};
    for (std::size_t k = 0; k < N_DEGREE; ++k) {
      x_cheb_nodes[k] = static_cast<InputType>(std::cos((2.0 * k + 1.0) * M_PI / (2.0 * N_DEGREE)));
    }

    std::array<OutputType, N_DEGREE> y_cheb_values{};
    for (std::size_t i = 0; i < N_DEGREE; ++i) {
      y_cheb_values[i] = F(map_to_domain(x_cheb_nodes[i]));
    }

    std::array<OutputType, N_DEGREE> newton_coeffs = bjorck_pereyra_constexpr(x_cheb_nodes, y_cheb_values);
    std::array<OutputType, N_DEGREE> monomial_coeffs = newton_to_monomial_constexpr(newton_coeffs, x_cheb_nodes);

    std::array<OutputType, N_DEGREE> current_coeffs = monomial_coeffs;
    for (std::size_t pass = 0; pass < ITERS; ++pass) {
      std::array<OutputType, N_DEGREE> r_cheb{};
      for (std::size_t i = 0; i < N_DEGREE; ++i) {
        InputType xi = x_cheb_nodes[i];
        OutputType p_val = horner(current_coeffs, xi);
        r_cheb[i] = y_cheb_values[i] - p_val;
      }
      std::array<OutputType, N_DEGREE> newton_r = bjorck_pereyra_constexpr(x_cheb_nodes, r_cheb);
      std::array<OutputType, N_DEGREE> mono_r = newton_to_monomial_constexpr(newton_r, x_cheb_nodes);

      for (std::size_t j = 0; j < N_DEGREE; ++j) {
        current_coeffs[j] += mono_r[j];
      }
    }
    return current_coeffs;
  }
};

// -----------------------------------------------------------------------------
// make_constexpr_func_eval: Full compile-time fitting API (C++20 only)
// Finds minimum N up to MaxN_val at compile-time and returns a ConstexprFuncEval
// with that specific N. This requires a helper struct to manage the
// recursive compile-time search for N.
// -----------------------------------------------------------------------------
#if __cplusplus >= 202002L

namespace internal {
template <std::size_t CurrentN, std::size_t MaxN_val, std::size_t NumEvalPoints_val, std::size_t Iters_compile_time,
          class Func, typename InputType, typename OutputType>
constexpr auto find_optimal_constexpr_eval_impl(Func F, InputType a, InputType b, double eps_val) {
  if constexpr (CurrentN > MaxN_val) {
    std::cout << "Warning: Did not converge to epsilon " << std::scientific << std::setprecision(4) << eps_val
        << " within MaxN = " << MaxN_val << ". Returning ConstexprFuncEval with degree " << MaxN_val << ".\n";
    return ConstexprFuncEval<Func, MaxN_val, Iters_compile_time>(F, a, b);
  } else {
    ConstexprFuncEval<Func, CurrentN, Iters_compile_time> current_evaluator(F, a, b);
    constexpr std::array<InputType, NumEvalPoints_val> eval_points = constexpr_linspace<
      InputType, NumEvalPoints_val>(a, b);

    double max_observed_error = 0.0;
    for (std::size_t i = 0; i < NumEvalPoints_val; ++i) {
      InputType pt = eval_points[i];
      OutputType actual_val = F(pt);
      OutputType poly_val = current_evaluator(pt);

      if (std::abs(actual_val) < std::numeric_limits<OutputType>::epsilon()) {
        double current_abs_error = std::abs(poly_val);
        if (current_abs_error > max_observed_error) {
          max_observed_error = current_abs_error;
        }
      } else {
        double current_abs_error = std::abs(1.0 - std::abs(poly_val / actual_val));
        if (current_abs_error > max_observed_error) {
          max_observed_error = current_abs_error;
        }
      }
    }

    if (max_observed_error <= eps_val) {
      std::cout << "Converged: Found min degree N = " << CurrentN
          << " (Max Error: " << std::scientific << std::setprecision(4) << max_observed_error
          << " <= Epsilon: " << eps_val << ")\n";
      return current_evaluator;
    } else {
      return find_optimal_constexpr_eval_impl<CurrentN + 1, MaxN_val, NumEvalPoints_val, Iters_compile_time>(
          F, a, b, eps_val);
    }
  }
}
} // namespace internal

template <double eps_val, std::size_t MaxN_val, std::size_t NumEvalPoints_val, std::size_t Iters_compile_time = 1, class
          Func>
constexpr auto make_constexpr_func_eval(Func F,
                                        typename function_traits<Func>::arg0_type a,
                                        typename function_traits<Func>::arg0_type b) {
  using InputType = typename function_traits<Func>::arg0_type;
  using OutputType = typename function_traits<Func>::result_type;

  static_assert(MaxN_val > 0, "Max polynomial degree for compile-time fitting must be positive.");
  static_assert(NumEvalPoints_val > 1, "Number of evaluation points for compile-time fitting must be greater than 1.");

  double validated_eps = eps_val;
  if (validated_eps < std::numeric_limits<double>::epsilon()) {
    if constexpr (std::is_floating_point_v<OutputType>) {
      if (validated_eps < std::numeric_limits<OutputType>::epsilon()) {
        std::cerr << "Warning: Requested epsilon " << validated_eps
            << " is less than machine epsilon for OutputType ("
            << std::numeric_limits<OutputType>::epsilon() << "). Clamping.\n";
        validated_eps = std::numeric_limits<OutputType>::epsilon();
      }
    } else if constexpr (std::is_same_v<OutputType, std::complex<float>>) {
      if (validated_eps < std::numeric_limits<float>::epsilon()) {
        std::cerr << "Warning: Requested epsilon " << validated_eps
            << " is less than machine epsilon for std::complex<float> ("
            << std::numeric_limits<float>::epsilon() << "). Clamping.\n";
        validated_eps = std::numeric_limits<float>::epsilon();
      }
    } else if constexpr (std::is_same_v<OutputType, std::complex<double>>) {
      if (validated_eps < std::numeric_limits<double>::epsilon()) {
        std::cerr << "Warning: Requested epsilon " << validated_eps
            << " is less than machine epsilon for std::complex<double> ("
            << std::numeric_limits<double>::epsilon() << "). Clamping.\n";
        validated_eps = std::numeric_limits<double>::epsilon();
      }
    }
  }

  return internal::find_optimal_constexpr_eval_impl<1, MaxN_val, NumEvalPoints_val, Iters_compile_time>(
      F, a, b, validated_eps);
}

template <std::size_t N_DEGREE, std::size_t Iters_compile_time = 1, class Func>
constexpr auto make_constexpr_fixed_degree_eval(Func F,
                                                typename function_traits<Func>::arg0_type a,
                                                typename function_traits<Func>::arg0_type b) {
  static_assert(N_DEGREE > 0, "Degree must be positive for compile-time fitting.");
  return ConstexprFuncEval<Func, N_DEGREE, Iters_compile_time>(F, a, b);
}

#endif // __cplusplus >= 202002L

} // namespace poly_eval