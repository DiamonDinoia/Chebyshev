// poly_eval_constexpr_impl.hpp - C++20 constexpr implementation details
#pragma once

#if __cplusplus >= 202002L

// No need to include "poly_eval.hpp" here, as it's included by poly_eval.hpp
// This file is meant to be included *by* poly_eval.hpp

namespace poly_eval {

// -----------------------------------------------------------------------------
// ConstexprFuncEval Implementation (C++20)
// -----------------------------------------------------------------------------

template <class Func, std::size_t N_DEGREE, std::size_t ITERS>
constexpr ConstexprFuncEval<Func, N_DEGREE, ITERS>::ConstexprFuncEval(Func F, InputType a, InputType b)
  : low(b - a), hi(b + a), coeffs_(initialize_coeffs(F, a, b)) {
}

template <class Func, std::size_t N_DEGREE, std::size_t ITERS>
constexpr typename ConstexprFuncEval<Func, N_DEGREE, ITERS>::OutputType
ConstexprFuncEval<Func, N_DEGREE, ITERS>::operator()(InputType pt) const noexcept {
  InputType xi = map_from_domain(pt);
  return horner(coeffs_, xi);
}

template <class Func, std::size_t N_DEGREE, std::size_t ITERS>
constexpr const std::array<typename ConstexprFuncEval<Func, N_DEGREE, ITERS>::OutputType, N_DEGREE> &
ConstexprFuncEval<Func, N_DEGREE, ITERS>::coeffs() const noexcept {
  return coeffs_;
}

template <class Func, std::size_t N_DEGREE, std::size_t ITERS>
constexpr typename ConstexprFuncEval<Func, N_DEGREE, ITERS>::InputType
ConstexprFuncEval<Func, N_DEGREE, ITERS>::map_to_domain(const InputType T_arg) const { return 0.5 * (low * T_arg + hi); }

template <class Func, std::size_t N_DEGREE, std::size_t ITERS>
constexpr typename ConstexprFuncEval<Func, N_DEGREE, ITERS>::InputType
ConstexprFuncEval<Func, N_DEGREE, ITERS>::map_from_domain(const InputType T_arg) const { return (2.0 * T_arg - hi) / low; }

template <class Func, std::size_t N_DEGREE, std::size_t ITERS>
constexpr typename ConstexprFuncEval<Func, N_DEGREE, ITERS>::OutputType
ConstexprFuncEval<Func, N_DEGREE, ITERS>::horner(const std::array<OutputType, N_DEGREE> &c, InputType x) noexcept {
  if constexpr (N_DEGREE == 0) {
    return static_cast<OutputType>(0.0);
  } else {
    return horner_forward_step<N_DEGREE, 0, OutputType, InputType>(c, x);
  }
}

template <class Func, std::size_t N_DEGREE, std::size_t ITERS>
constexpr std::array<typename ConstexprFuncEval<Func, N_DEGREE, ITERS>::OutputType, N_DEGREE>
ConstexprFuncEval<Func, N_DEGREE, ITERS>::bjorck_pereyra_constexpr(const std::array<InputType, N_DEGREE> &x,
                                                                 const std::array<OutputType, N_DEGREE> &y) noexcept {
  std::array<OutputType, N_DEGREE> a = y;
  for (std::size_t k = 0; k < N_DEGREE - 1; ++k) {
    for (std::size_t i = N_DEGREE - 1; i >= k + 1; --i) {
      // In a real application, consider a compile-time error or a fallback for division by zero.
      a[i] = (a[i] - a[i - 1]) / static_cast<OutputType>(x[i] - x[i - k - 1]);
    }
  }
  return a;
}

template <class Func, std::size_t N_DEGREE, std::size_t ITERS>
constexpr std::array<typename ConstexprFuncEval<Func, N_DEGREE, ITERS>::OutputType, N_DEGREE>
ConstexprFuncEval<Func, N_DEGREE, ITERS>::newton_to_monomial_constexpr(const std::array<OutputType, N_DEGREE> &alpha,
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

template <class Func, std::size_t N_DEGREE, std::size_t ITERS>
constexpr std::array<typename ConstexprFuncEval<Func, N_DEGREE, ITERS>::OutputType, N_DEGREE>
ConstexprFuncEval<Func, N_DEGREE, ITERS>::initialize_coeffs(Func F, InputType a, InputType b) {
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

// -----------------------------------------------------------------------------
// make_func_eval API implementations (C++20 for compile-time eps)
// -----------------------------------------------------------------------------

template <double eps_val, std::size_t MaxN_val, std::size_t NumEvalPoints_val, std::size_t Iters_compile_time, class
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


// -----------------------------------------------------------------------------
// make_constexpr_func_eval API implementation (Full Compile-Time)
// -----------------------------------------------------------------------------

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

template <double eps_val, std::size_t MaxN_val, std::size_t NumEvalPoints_val, std::size_t Iters_compile_time, class
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

template <std::size_t N_DEGREE, std::size_t Iters_compile_time, class Func>
constexpr auto make_constexpr_fixed_degree_eval(Func F,
                                                typename function_traits<Func>::arg0_type a,
                                                typename function_traits<Func>::arg0_type b) {
  static_assert(N_DEGREE > 0, "Degree must be positive for compile-time fitting.");
  return ConstexprFuncEval<Func, N_DEGREE, Iters_compile_time>(F, a, b);
}

} // namespace poly_eval

#endif // __cplusplus >= 202002L