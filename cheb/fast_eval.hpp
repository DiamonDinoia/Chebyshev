#pragma once

#include <array>
#include <cmath>
#include <experimental/mdspan>
#include <functional>
#include <iostream>
#include <type_traits>
#include <vector>

#include "macros.h"

#if __cplusplus < 202002L
namespace std {
template <typename T> using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

constexpr bool is_constant_evaluated() noexcept {
  return false; // Always returns false in pre-C++20 code
}
} // namespace std
#endif

namespace poly_eval {

template <typename T> struct function_traits : function_traits<decltype(&T::operator())> {};

template <typename T, typename = void> struct is_tuple_like : std::false_type {};

template <typename T>
struct is_tuple_like<T, std::void_t<decltype(std::tuple_size_v<std::remove_cvref_t<T>>)>> : std::true_type {};

#if __cpp_concepts >= 201907L
template <typename T>
concept tuple_like = is_tuple_like<T>::value;
#endif

// Convenience: size-or-zero that never hard-errors
template <typename T, typename = void> struct tuple_size_or_zero : std::integral_constant<std::size_t, 0> {};

template <typename T>
struct tuple_size_or_zero<T, std::void_t<decltype(std::tuple_size<std::remove_cvref_t<T>>::value)>>
    : std::integral_constant<std::size_t, std::tuple_size<std::remove_cvref_t<T>>::value> {};

// -----------------------------------------------------------------------------
// Buffer: Conditional type alias for std::vector or std::array
// -----------------------------------------------------------------------------
template <typename T, std::size_t N_compile_time_val>
using Buffer = std::conditional_t<N_compile_time_val == 0, std::vector<T>, std::array<T, N_compile_time_val>>;

// -----------------------------------------------------------------------------
// Forward declarations for FuncEvalMany
template <typename... EvalTypes> class FuncEvalMany;
// Forward declaration for FuncEval

// -----------------------------------------------------------------------------
// FuncEval: monomial least-squares fit using Chebyshev sampling
// (Runtime or Fixed-Size Compile-Time Storage, but fitting is runtime)
// -----------------------------------------------------------------------------
template <class Func, std::size_t N_compile_time = 0, std::size_t Iters_compile_time = 1> class FuncEval {
public:
  using InputType = typename function_traits<Func>::arg0_type;
  using OutputType = typename function_traits<Func>::result_type;

  static constexpr std::size_t kDegreeCompileTime = N_compile_time;
  static constexpr std::size_t kItersCompileTime = Iters_compile_time;

  template <std::size_t CurrentN = N_compile_time, typename = std::enable_if_t<CurrentN != 0>>
  C20CONSTEXPR FuncEval(Func F, InputType a, InputType b, const InputType *pts = nullptr);

  template <std::size_t CurrentN = N_compile_time, typename = std::enable_if_t<CurrentN == 0>>
  C20CONSTEXPR FuncEval(Func F, int n, InputType a, InputType b, const InputType *pts = nullptr);

  constexpr OutputType operator()(InputType pt) const noexcept;

  template <bool pts_aligned = false, bool out_aligned = false>
  constexpr void operator()(const InputType *pts, OutputType *out, std::size_t num_points) const noexcept;

  C20CONSTEXPR const Buffer<OutputType, N_compile_time> &coeffs() const noexcept;

private:
  const int n_terms;
  const InputType low, hi;
  Buffer<OutputType, N_compile_time> monomials;

  C20CONSTEXPR void initialize_monomials(Func F, const InputType *pts);

  template <class T> ALWAYS_INLINE constexpr T map_to_domain(T T_arg) const noexcept;
  template <class T> ALWAYS_INLINE constexpr T map_from_domain(T T_arg) const noexcept;

  // Evaluate multiple points using SIMD with unrolling
  template <int OuterUnrollFactor, bool pts_aligned, bool out_aligned>
  constexpr void horner_polyeval(const InputType *pts, OutputType *out, std::size_t num_points) const noexcept;

  template <int OuterUnrollFactor, bool pts_aligned, bool out_aligned>
  NO_INLINE constexpr void no_inline_horner_polyeval(const InputType *pts, OutputType *out,
                                           std::size_t num_points) const noexcept;

  C20CONSTEXPR static Buffer<OutputType, N_compile_time> bjorck_pereyra(const Buffer<InputType, N_compile_time> &x,
                                                                        const Buffer<OutputType, N_compile_time> &y);

  C20CONSTEXPR static Buffer<OutputType, N_compile_time>
  newton_to_monomial(const Buffer<OutputType, N_compile_time> &alpha, const Buffer<InputType, N_compile_time> &nodes);

  C20CONSTEXPR void refine(const Buffer<InputType, N_compile_time> &x_cheb_,
                           const Buffer<OutputType, N_compile_time> &y_cheb_);

  // Friend declaration for FuncEvalMany to access private members
  template <typename... EvalTypes> friend class FuncEvalMany;
};

// FuncEvalMany: evaluates multiple FuncEval instances
// Supports both compile-time fixed degree and runtime degree
// FuncEvalMany: evaluates multiple FuncEval instances without storing EvalTypes
template <typename... EvalTypes> class FuncEvalMany {
  static_assert(sizeof...(EvalTypes) > 0, "At least one FuncEval is required");

  using FirstEval = std::tuple_element_t<0, std::tuple<EvalTypes...>>;
  using InputType = typename FirstEval::InputType;
  using OutputType = typename FirstEval::OutputType;

  static constexpr std::size_t kF = sizeof...(EvalTypes);
  static constexpr std::size_t deg_max_ctime_ = std::max({EvalTypes::kDegreeCompileTime...});

  // Actual degree in use (runtime if deg_max_ctime_ == 0)
  std::size_t deg_max_ = deg_max_ctime_;

  // Contiguous storage for coefficients
  Buffer<OutputType, kF * deg_max_ctime_> coeff_storage_;
  static constexpr std::size_t dyn_extent = std::experimental::dynamic_extent;
  using Extents = std::experimental::extents<std::size_t, kF, (deg_max_ctime_ != 0 ? deg_max_ctime_ : dyn_extent)>;
  std::experimental::mdspan<OutputType, Extents> coeffs_;

  // Contiguous mapping parameters
  std::array<InputType, kF> low_;
  std::array<InputType, kF> hi_;

public:
  // Constructor: copies low/hi, determines runtime degree, allocates and copies coefficients
  explicit FuncEvalMany(const EvalTypes &...evals)
      : low_{evals.low...}, hi_{evals.hi...}, coeffs_{nullptr, kF, deg_max_ctime_} {
    if constexpr (deg_max_ctime_ == 0) {
      deg_max_ = std::max({evals.n_terms...});
      coeff_storage_.assign(kF * deg_max_, OutputType{});
      coeffs_ = std::experimental::mdspan<OutputType, Extents>(coeff_storage_.data(), kF, deg_max_);
    } else {
      coeffs_ = std::experimental::mdspan<OutputType, Extents>(coeff_storage_.data(), kF, deg_max_ctime_);
    }

    copy_coeffs<0>(evals...);
  }

  [[nodiscard]] std::size_t size() const noexcept { return kF; }
  [[nodiscard]] std::size_t degree() const noexcept { return deg_max_; }

  // Evaluate the idx-th polynomial at x
  [[nodiscard]] OutputType operator()(InputType x, std::size_t idx) const noexcept {
    InputType xu = (2 * x - hi_[idx]) * low_[idx];
    auto row = std::experimental::submdspan(coeffs_, idx, std::experimental::full_extent);
    OutputType acc = row[deg_max_ - 1];
    for (std::size_t k = deg_max_ - 1; k-- > 0;) {
      acc = std::fma(acc, xu, row[k]);
    }
    return acc;
  }

  // Evaluate all polynomials at the same input
  [[nodiscard]] std::array<OutputType, kF> operator()(InputType x) const noexcept {
    std::array<OutputType, kF> results;
    for (std::size_t i = 0; i < kF; ++i) {
      InputType xu = (2 * x - hi_[i]) * low_[i];
      auto row = std::experimental::submdspan(coeffs_, i, std::experimental::full_extent);
      OutputType acc = row[deg_max_ - 1];
      for (std::size_t k = deg_max_ - 1; k-- > 0;) {
        acc = std::fma(acc, xu, row[k]);
      }
      results[i] = acc;
    }
    return results;
  }

  // Evaluate with tuple of inputs: each element maps to corresponding polynomial
  template <typename... Ts>
  [[nodiscard]] std::array<OutputType, kF> operator()(const std::tuple<Ts...> &inputs) const noexcept {
    static_assert(sizeof...(Ts) == kF, "Tuple size must match number of functions");
    std::array<InputType, kF> arr{};
    std::apply([&](auto &&...elems) { arr = {static_cast<InputType>(elems)...}; }, inputs);
    return operator()(arr);
  }

  // Evaluate with array of inputs
  [[nodiscard]] std::array<OutputType, kF> operator()(const std::array<InputType, kF> &inputs) const noexcept {
    std::array<OutputType, kF> results;
    for (std::size_t i = 0; i < kF; ++i) {
      InputType xu = (2 * inputs[i] - hi_[i]) * low_[i];
      auto row = std::experimental::submdspan(coeffs_, i, std::experimental::full_extent);
      OutputType acc = row[deg_max_ - 1];
      for (std::size_t k = deg_max_ - 1; k-- > 0;) {
        acc = std::fma(acc, xu, row[k]);
      }
      results[i] = acc;
    }
    return results;
  }

  // Variadic inputs call
  template <typename... Ts>
  [[nodiscard]] std::array<OutputType, kF> operator()(InputType first, Ts... rest) const noexcept {
    static_assert(1 + sizeof...(Ts) == kF, "Number of arguments must match number of functions");
    return operator()(std::array<InputType, kF>{first, static_cast<InputType>(rest)...});
  }

private:
  // Recursive helper: copy and pad coefficients for polynomial I
  template <std::size_t I, typename FE, typename... Rest> void copy_coeffs(const FE &fe, const Rest &...rest) {
    for (std::size_t k = 0; k < fe.n_terms; ++k) {
      coeffs_(I, k) = fe.monomials[k];
    }
    for (std::size_t k = fe.n_terms; k < deg_max_; ++k) {
      coeffs_(I, k) = OutputType{0};
    }
    if constexpr (I + 1 < kF) {
      copy_coeffs<I + 1>(rest...);
    }
  }
};

// 1) Compile-time degree only
template <std::size_t N_compile_time, std::size_t Iters_compile_time = 1, class Func>
C20CONSTEXPR auto make_func_eval(Func F, typename function_traits<Func>::arg0_type a,
                                 typename function_traits<Func>::arg0_type b,
                                 const typename function_traits<Func>::arg0_type *pts = nullptr);

// 2) Runtime degree (N_compile_time==0 internally)
template <std::size_t Iters_compile_time = 1, class Func>
C20CONSTEXPR auto make_func_eval(Func F, int n, typename function_traits<Func>::arg0_type a,
                                 typename function_traits<Func>::arg0_type b,
                                 const typename function_traits<Func>::arg0_type *pts = nullptr);

// 3) C++17-compatible: runtime error tolerance
template <std::size_t MaxN_val, std::size_t NumEvalPoints_val, std::size_t Iters_compile_time = 1, class Func>
C20CONSTEXPR auto make_func_eval(Func F, double eps, typename function_traits<Func>::arg0_type a,
                                 typename function_traits<Func>::arg0_type b);

#if __cplusplus >= 202002L
// 4) C++20: compile-time error tolerance
template <double eps_val, std::size_t MaxN_val, std::size_t NumEvalPoints_val, std::size_t Iters_compile_time = 1,
          class Func>
constexpr auto make_func_eval(Func F, typename function_traits<Func>::arg0_type a,
                              typename function_traits<Func>::arg0_type b);
#endif

template <typename... EvalTypes> C20CONSTEXPR FuncEvalMany<EvalTypes...> make_func_eval(EvalTypes... evals) noexcept;

} // namespace poly_eval

// Include implementations
// ReSharper disable once CppUnusedIncludeDirective
#include "fast_eval_impl.hpp"