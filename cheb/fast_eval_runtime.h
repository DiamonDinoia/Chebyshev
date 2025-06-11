// poly_eval_runtime_impl.hpp - C++17 compatible runtime implementation details
#pragma once
#include <cassert>
#include <iomanip>
#include <iostream>

// No need to include "poly_eval.hpp" here, as it's included by poly_eval.hpp
// This file is meant to be included *by* poly_eval.hpp

namespace poly_eval {


namespace detail {
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
}

// -----------------------------------------------------------------------------
// FuncEval Implementation (Runtime)
// -----------------------------------------------------------------------------

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
template <std::size_t CurrentN, typename>
FuncEval<Func, N_compile_time, Iters_compile_time>::FuncEval(Func F, int n, InputType a, InputType b)
  : deg_(n), low(b - a), hi(b + a) {
  assert(deg_ > 0 && "Polynomial degree must be positive");
  coeffs_.resize(deg_);
  initialize_coeffs(F);
}

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
template <std::size_t CurrentN, typename>
FuncEval<Func, N_compile_time, Iters_compile_time>::FuncEval(Func F, InputType a, InputType b)
  : deg_(static_cast<int>(CurrentN)), low(b - a), hi(b + a) {
  assert(deg_ > 0 && "Polynomial degree must be positive (template N > 0)");
  initialize_coeffs(F);
}

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
typename FuncEval<Func, N_compile_time, Iters_compile_time>::OutputType
FuncEval<Func, N_compile_time, Iters_compile_time>::operator()(InputType pt) const noexcept {
  InputType xi = map_from_domain(pt);
  return horner(coeffs_, xi);
}

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
void FuncEval<Func, N_compile_time, Iters_compile_time>::operator()(InputType *pts, OutputType *out,
                                                                    int num_points) const noexcept {
  for (int i = 0; i < num_points; ++i) {
    out[i] = (*this)(pts[i]);
  }
}

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
const Buffer<typename FuncEval<Func, N_compile_time, Iters_compile_time>::OutputType, N_compile_time> &
FuncEval<Func, N_compile_time, Iters_compile_time>::coeffs() const noexcept {
  return coeffs_;
}

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
void FuncEval<Func, N_compile_time, Iters_compile_time>::initialize_coeffs(Func F) {
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

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
template <class T> constexpr T FuncEval<Func, N_compile_time, Iters_compile_time>::map_to_domain(const T T_arg) const {
  return static_cast<T>(0.5 * (low * T_arg + hi));
}

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
template <class T> constexpr T FuncEval<Func, N_compile_time,
                                        Iters_compile_time>::map_from_domain(const T T_arg) const {
  return static_cast<T>((2.0 * T_arg - hi) / low);
}

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
typename FuncEval<Func, N_compile_time, Iters_compile_time>::OutputType
FuncEval<Func, N_compile_time, Iters_compile_time>::horner(const Buffer<OutputType, N_compile_time> &c,
                                                           InputType x) noexcept {
  if (c.empty()) {
    return static_cast<OutputType>(0.0);
  }
  OutputType acc = c[c.size() - 1]; // Start with the highest degree coefficient
  for (int k = static_cast<int>(c.size()) - 2; k >= 0; --k) {
    acc = acc * x + c[k];
  }
  return acc;
}

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
std::vector<typename FuncEval<Func, N_compile_time, Iters_compile_time>::OutputType>
FuncEval<Func, N_compile_time, Iters_compile_time>::bjorck_pereyra(const std::vector<InputType> &x,
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

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
std::vector<typename FuncEval<Func, N_compile_time, Iters_compile_time>::OutputType>
FuncEval<Func, N_compile_time, Iters_compile_time>::newton_to_monomial(const std::vector<OutputType> &alpha,
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

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
void FuncEval<Func, N_compile_time, Iters_compile_time>::refine_via_bjorck_pereyra(
    const std::vector<InputType> &x_cheb_,
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

// -----------------------------------------------------------------------------
// make_func_eval API implementations (Runtime, C++17 compatible)
// -----------------------------------------------------------------------------

template <std::size_t N_compile_time, std::size_t Iters_compile_time, class Func>
auto make_func_eval(Func F,
                    typename function_traits<Func>::arg0_type a,
                    typename function_traits<Func>::arg0_type b) {
  return FuncEval<Func, N_compile_time, Iters_compile_time>(F, a, b);
}

template <std::size_t Iters_compile_time, class Func>
auto make_func_eval(Func F, int n,
                    typename function_traits<Func>::arg0_type a,
                    typename function_traits<Func>::arg0_type b) {
  return FuncEval<Func, 0, Iters_compile_time>(F, n, a, b);
}

// C++17 compatible API for finding minimum N for a given error tolerance (runtime eps)
template <std::size_t MaxN_val, std::size_t NumEvalPoints_val, std::size_t Iters_compile_time, class Func>
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

  std::vector<InputType> eval_points = poly_eval::detail::linspace(a, b, static_cast<int>(NumEvalPoints_val));

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

} // namespace poly_eval