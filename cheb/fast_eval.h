// poly_eval.hpp - Public Interface and Includes
#pragma once

#include <array>
#include <vector>
#include <cmath>
#include <complex>
#include <utility>
#include <functional>
#include <type_traits>

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
  FuncEval(Func F, int n, InputType a, InputType b);

  template <std::size_t CurrentN = N_compile_time,
            typename = std::enable_if_t<CurrentN != 0>>
  FuncEval(Func F, InputType a, InputType b);

  OutputType operator()(InputType pt) const noexcept;

  void operator()(InputType* pts, OutputType* out, int num_points) const noexcept;

  const Buffer<OutputType, N_compile_time> &coeffs() const noexcept;

private:
  int deg_;
  const InputType low, hi;
  Buffer<OutputType, N_compile_time> coeffs_;

  void initialize_coeffs(Func F);

  template <class T> constexpr T map_to_domain(T T_arg) const;
  template <class T> constexpr T map_from_domain(T T_arg) const;

  static OutputType horner(const Buffer<OutputType, N_compile_time> &c, InputType x) noexcept;

  std::vector<OutputType> bjorck_pereyra(const std::vector<InputType> &x,
                                         const std::vector<OutputType> &y) const;

  static std::vector<OutputType> newton_to_monomial(const std::vector<OutputType> &alpha,
                                                    const std::vector<InputType> &nodes);

  void refine_via_bjorck_pereyra(const std::vector<InputType> &x_cheb_,
                                 const std::vector<OutputType> &y_cheb_);
};

// -----------------------------------------------------------------------------
// Unified make_func_eval API (for runtime or fixed-size, runtime-fitted evaluation)
// -----------------------------------------------------------------------------

// Overload 1: For COMPILE-TIME degree N_compile_time (> 0)
template <std::size_t N_compile_time, std::size_t Iters_compile_time = 1, class Func>
auto make_func_eval(Func F,
                    typename function_traits<Func>::arg0_type a,
                    typename function_traits<Func>::arg0_type b);

// Overload 2: For RUNTIME degree 'n' (N_compile_time = 0 internally)
template <std::size_t Iters_compile_time = 1, class Func>
auto make_func_eval(Func F, int n,
                    typename function_traits<Func>::arg0_type a,
                    typename function_traits<Func>::arg0_type b);


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

  constexpr ConstexprFuncEval(Func F, InputType a, InputType b);

  constexpr OutputType operator()(InputType pt) const noexcept;

  constexpr void operator()(InputType* pts, OutputType* out, int num_points) const noexcept;

  constexpr const std::array<OutputType, N_DEGREE> &coeffs() const noexcept;


private:
  const InputType low, hi;
  std::array<OutputType, N_DEGREE> coeffs_;

  constexpr InputType map_to_domain(InputType T_arg) const;
  constexpr InputType map_from_domain(InputType T_arg) const;

  static constexpr OutputType horner(const std::array<OutputType, N_DEGREE> &c, InputType x) noexcept;

  static constexpr std::array<OutputType, N_DEGREE>
  bjorck_pereyra_constexpr(const std::array<InputType, N_DEGREE> &x,
                           const std::array<OutputType, N_DEGREE> &y) noexcept;

  static constexpr std::array<OutputType, N_DEGREE>
  newton_to_monomial_constexpr(const std::array<OutputType, N_DEGREE> &alpha,
                               const std::array<InputType, N_DEGREE> &nodes) noexcept;

  constexpr std::array<OutputType, N_DEGREE> initialize_coeffs(Func F, InputType a, InputType b);
};

// -----------------------------------------------------------------------------
// C++17 Compatible make_func_eval for runtime error tolerance
// -----------------------------------------------------------------------------
template <std::size_t MaxN_val, std::size_t NumEvalPoints_val, std::size_t Iters_compile_time = 1, class Func>
auto make_func_eval(Func F, double eps, // eps as a runtime parameter
                    typename function_traits<Func>::arg0_type a,
                    typename function_traits<Func>::arg0_type b);

#if __cplusplus >= 202002L
// -----------------------------------------------------------------------------
// C++20 make_func_eval for compile-time error tolerance (runtime fitting)
// -----------------------------------------------------------------------------
template <double eps_val, std::size_t MaxN_val, std::size_t NumEvalPoints_val, std::size_t Iters_compile_time = 1, class
          Func>
auto make_func_eval(Func F,
                    typename function_traits<Func>::arg0_type a,
                    typename function_traits<Func>::arg0_type b);

// -----------------------------------------------------------------------------
// C++20 make_constexpr_func_eval: Full compile-time fitting API
// -----------------------------------------------------------------------------
template <double eps_val, std::size_t MaxN_val, std::size_t NumEvalPoints_val, std::size_t Iters_compile_time = 1, class
          Func>
constexpr auto make_constexpr_func_eval(Func F,
                                        typename function_traits<Func>::arg0_type a,
                                        typename function_traits<Func>::arg0_type b);

// -----------------------------------------------------------------------------
// C++20 make_constexpr_fixed_degree_eval: Compile-time fitting for a fixed degree
// -----------------------------------------------------------------------------
template <std::size_t N_DEGREE, std::size_t Iters_compile_time = 1, class Func>
constexpr auto make_constexpr_func_eval(Func F,
                                                typename function_traits<Func>::arg0_type a,
                                                typename function_traits<Func>::arg0_type b);
#endif // __cplusplus >= 202002L

} // namespace poly_eval

// Include implementations
#include "fast_eval_runtime.h"
#if __cplusplus >= 202002L
#include "fast_eval_compile_time.h"
#endif