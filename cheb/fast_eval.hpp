#pragma once

#include <array>
#include <cmath>
#include <functional>
#include <type_traits>
#include <vector>

#include "macros.h"

namespace poly_eval {

// -----------------------------------------------------------------------------
// function_traits: Helper to deduce input and output types from a callable
// -----------------------------------------------------------------------------
template <typename T> struct function_traits;

template <typename R, typename Arg> struct function_traits<R (*)(Arg)> {
  using result_type = R;
  using arg0_type = Arg;
};

template <typename R, typename Arg> struct function_traits<std::function<R(Arg)>> {
  using result_type = R;
  using arg0_type = Arg;
};

template <typename F, typename R, typename Arg> struct function_traits<R (F::*)(Arg) const> {
  using result_type = R;
  using arg0_type = Arg;
};

template <typename F, typename R, typename Arg> struct function_traits<R (F::*)(Arg)> {
  using result_type = R;
  using arg0_type = Arg;
};

template <typename T> struct function_traits : function_traits<decltype(&T::operator())> {};

// -----------------------------------------------------------------------------
// Buffer: Conditional type alias for std::vector or std::array
// -----------------------------------------------------------------------------
template <typename T, std::size_t N_compile_time_val>
using Buffer = std::conditional_t<N_compile_time_val == 0, std::vector<T>, std::array<T, N_compile_time_val>>;

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

  // template <std::size_t CurrentN = N_compile_time,
  //           typename = std::enable_if_t<CurrentN == 0>>
  // C20CONSTEXPR FuncEval(Func F, int n, InputType a, InputType b);
  //
  // template <std::size_t CurrentN = N_compile_time,
  //           typename = std::enable_if_t<CurrentN != 0>>
  // C20CONSTEXPR FuncEval(Func F, InputType a, InputType b);

  template <std::size_t CurrentN = N_compile_time, typename = std::enable_if_t<CurrentN != 0>>
  C20CONSTEXPR FuncEval(Func F, InputType a, InputType b, const InputType *pts = nullptr);

  template <std::size_t CurrentN = N_compile_time, typename = std::enable_if_t<CurrentN == 0>>
  C20CONSTEXPR FuncEval(Func F, int n, InputType a, InputType b, const InputType *pts = nullptr);

  C20CONSTEXPR OutputType operator()(InputType pt) const noexcept;

  template <bool pts_aligned = false, bool out_aligned = false>
  void operator()(const InputType *pts, OutputType *out, std::size_t num_points) const noexcept;

  C20CONSTEXPR const Buffer<OutputType, N_compile_time> &coeffs() const noexcept;

private:
  const int n_terms;
  const InputType low, hi;
  Buffer<OutputType, N_compile_time> monomials;

  C20CONSTEXPR void initialize_monomials(Func F, const InputType *pts);

  template <class T> ALWAYS_INLINE constexpr T map_to_domain(T T_arg) const;
  template <class T> ALWAYS_INLINE constexpr T map_from_domain(T T_arg) const;

  // Refactored declaration for horner
  constexpr static OutputType horner(const OutputType *c_ptr, std::size_t c_size, InputType x) noexcept;

  template <std::size_t N_total, std::size_t current_idx>
  static constexpr OutputType horner(const OutputType *c_ptr, InputType x) noexcept;

  template <int K_Current, int K_Target, int OuterUnrollFactor, class VecInputType, class VecOutputType>
  void horner(VecInputType *pt_batches, VecOutputType *acc_batches) const noexcept;

  // Evaluate multiple points using SIMD with unrolling
  template <int OuterUnrollFactor, bool pts_aligned, bool out_aligned>
  void horner_polyeval(const InputType *pts, OutputType *out, std::size_t num_points) const noexcept;

  template <int OuterUnrollFactor, bool pts_aligned, bool out_aligned>
  void no_inline_horner_polyeval(const InputType *pts, OutputType *out, std::size_t num_points) const noexcept;

  C20CONSTEXPR static std::vector<OutputType> bjorck_pereyra(const std::vector<InputType> &x,
                                                             const std::vector<OutputType> &y);

  C20CONSTEXPR static std::vector<OutputType> newton_to_monomial(const std::vector<OutputType> &alpha,
                                                                 const std::vector<InputType> &nodes);

  C20CONSTEXPR void refine(const std::vector<InputType> &x_cheb_, const std::vector<OutputType> &y_cheb_);
};

// -----------------------------------------------------------------------------
// FuncEvalGroup: Groups multiple FuncEval instances and provides
// - coeffs(): tuple of each FuncEval's coefficient buffer
// - operator()(pt): evaluates all functions at pt, returning std::array<OutputType, N>
// -----------------------------------------------------------------------------

template <typename... EvalTypes> class FuncEvalMany {
  // Deduce common InputType and OutputType
  using FirstEval = std::tuple_element_t<0, std::tuple<EvalTypes...>>;

public:
  using InputType = typename FirstEval::arg0_type;
  using OutputType = typename FirstEval::result_type;

  /// Construct from pre-built FuncEval objects
  C20CONSTEXPR explicit FuncEvalMany(EvalTypes... evals) : evals_{std::move(evals)...} {}

  /// Number of functions in the group
  [[nodiscard]] constexpr std::size_t size() const noexcept { return sizeof...(EvalTypes); }

  /// Return a tuple of each FuncEval::coeffs()
  constexpr auto coeffs() const { return coeffs_impl(std::make_index_sequence<sizeof...(EvalTypes)>{}); }

  /// Evaluate all functions at a single point, returning std::array
  std::array<OutputType, sizeof...(EvalTypes)> C20CONSTEXPR operator()(InputType pt) const noexcept {
    return eval_at_impl(pt, std::make_index_sequence<sizeof...(EvalTypes)>{});
  }

private:
  std::tuple<EvalTypes...> evals_;

  // Helper to evaluate all at once into std::array
  template <std::size_t... I>
  constexpr std::array<OutputType, sizeof...(EvalTypes)> eval_at_impl(InputType pt,
                                                                      std::index_sequence<I...>) const noexcept {
    return {{std::get<I>(evals_)(pt)...}};
  }

  // Helper to gather coeffs into a tuple
  template <std::size_t... I> auto coeffs_impl(std::index_sequence<I...>) const {
    return std::make_tuple(std::get<I>(evals_).coeffs()...);
  }

  static_assert((std::is_same_v<typename EvalTypes::arg0_type, InputType> && ...),
                "All FuncEval types must have the same InputType");
  static_assert((std::is_same_v<typename EvalTypes::result_type, OutputType> && ...),
                "All FuncEval types must have the same OutputType");
  static_assert(sizeof...(EvalTypes) > 0, "At least one FuncEval is required");
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

template <typename... EvalTypes> C20CONSTEXPR FuncEvalMany<EvalTypes...> make_func_eval_group(EvalTypes... evals) {
  return FuncEvalGroup<EvalTypes...>(std::move(evals)...);
}
} // namespace poly_eval

// Include implementations
// ReSharper disable once CppUnusedIncludeDirective
#include "fast_eval_impl.hpp"