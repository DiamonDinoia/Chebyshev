// poly_eval.hpp - Public Interface and Includes
#pragma once

#include <array>
#include <vector>
#include <cmath>
#include <functional>
#include <type_traits>

#include "macros.h"


namespace poly_eval {

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
  C20CONSTEXPR FuncEval(Func F, int n, InputType a, InputType b);

  template <std::size_t CurrentN = N_compile_time,
            typename = std::enable_if_t<CurrentN != 0>>
  C20CONSTEXPR FuncEval(Func F, InputType a, InputType b);

  C20CONSTEXPR OutputType operator()(InputType pt) const noexcept;

  template <bool pts_aligned = false, bool out_aligned = false>
  void operator()(InputType *pts, OutputType *out, std::size_t num_points) const noexcept;

  C20CONSTEXPR const Buffer<OutputType, N_compile_time> &coeffs() const noexcept;

private:
  int deg_;
  const InputType low, hi;
  Buffer<OutputType, N_compile_time> coeffs_;

  C20CONSTEXPR void initialize_coeffs(Func F);

  template <class T> constexpr T map_to_domain(T T_arg) const;
  template <class T> constexpr T map_from_domain(T T_arg) const;

  // Refactored declaration for horner
  constexpr static OutputType horner(const OutputType *c_ptr, std::size_t c_size, InputType x) noexcept;

  template <std::size_t N_total, std::size_t current_idx>
  static constexpr OutputType horner(const OutputType *c_ptr, InputType x) noexcept;

  // Evaluate multiple points using SIMD with unrolling
  template <int OuterUnrollFactor, bool pts_aligned, bool out_aligned>
  void process_polynomial_horner(InputType *pts, OutputType *out, std::size_t num_points) const noexcept;


  C20CONSTEXPR static std::vector<OutputType> bjorck_pereyra(const std::vector<InputType> &x,
                                                             const std::vector<OutputType> &y);

  C20CONSTEXPR static std::vector<OutputType> newton_to_monomial(const std::vector<OutputType> &alpha,
                                                                 const std::vector<InputType> &nodes);

  C20CONSTEXPR void refine_via_bjorck_pereyra(const std::vector<InputType> &x_cheb_,
                                              const std::vector<OutputType> &y_cheb_);
};

// -----------------------------------------------------------------------------
// Unified make_func_eval API (for runtime or fixed-size, runtime-fitted evaluation)
// -----------------------------------------------------------------------------

// Overload 1: For COMPILE-TIME degree N_compile_time (> 0)
template <std::size_t N_compile_time, std::size_t Iters_compile_time = 1, class Func>
C20CONSTEXPR auto make_func_eval(Func F,
                                 typename function_traits<Func>::arg0_type a,
                                 typename function_traits<Func>::arg0_type b);

// Overload 2: For RUNTIME degree 'n' (N_compile_time = 0 internally)
template <std::size_t Iters_compile_time = 1, class Func>
C20CONSTEXPR auto make_func_eval(Func F, int n,
                                 typename function_traits<Func>::arg0_type a,
                                 typename function_traits<Func>::arg0_type b);

// -----------------------------------------------------------------------------
// C++17 Compatible make_func_eval for runtime error tolerance
// -----------------------------------------------------------------------------
template <std::size_t MaxN_val, std::size_t NumEvalPoints_val, std::size_t Iters_compile_time = 1, class Func>
C20CONSTEXPR auto make_func_eval(Func F, double eps, // eps as a runtime parameter
                                 typename function_traits<Func>::arg0_type a,
                                 typename function_traits<Func>::arg0_type b);

#if __cplusplus >= 202002L
// -----------------------------------------------------------------------------
// C++20 make_func_eval for compile-time error tolerance (runtime fitting)
// -----------------------------------------------------------------------------
template <double eps_val, std::size_t MaxN_val, std::size_t NumEvalPoints_val, std::size_t Iters_compile_time = 1, class
          Func>
constexpr auto make_func_eval(Func F,
                              typename function_traits<Func>::arg0_type a,
                              typename function_traits<Func>::arg0_type b);

#endif // __cplusplus >= 202002L

} // namespace poly_eval

// Include implementations
// ReSharper disable once CppUnusedIncludeDirective
#include "fast_eval_impl.hpp"