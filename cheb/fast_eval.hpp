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

  C20CONSTEXPR OutputType operator()(InputType pt) const noexcept;

  template <bool pts_aligned = false, bool out_aligned = false>
  void operator()(const InputType *pts, OutputType *out, std::size_t num_points) const noexcept;

  C20CONSTEXPR const Buffer<OutputType, N_compile_time> &coeffs() const noexcept;

private:
  const int n_terms;
  const InputType low, hi;
  Buffer<OutputType, N_compile_time> monomials;

  C20CONSTEXPR void initialize_monomials(Func F, const InputType *pts);

  template <class T> ALWAYS_INLINE constexpr T map_to_domain(T T_arg) const noexcept;
  template <class T> ALWAYS_INLINE constexpr T map_from_domain(T T_arg) const noexcept;

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

  C20CONSTEXPR static Buffer<OutputType, N_compile_time>
  bjorck_pereyra(const Buffer<InputType, N_compile_time>& x,
                 const Buffer<OutputType, N_compile_time>& y);

  C20CONSTEXPR static Buffer<OutputType, N_compile_time>
  newton_to_monomial(const Buffer<OutputType, N_compile_time>& alpha,
                     const Buffer<InputType, N_compile_time>& nodes);

  C20CONSTEXPR void
  refine(const Buffer<InputType, N_compile_time>& x_cheb_,
         const Buffer<OutputType, N_compile_time>& y_cheb_);

  // Friend declaration for FuncEvalMany to access private members
  template <typename... EvalTypes> friend class FuncEvalMany;
};

template <typename... EvalTypes>
class FuncEvalMany {
  static_assert(sizeof...(EvalTypes) > 0, "At least one FuncEval is required");
  using FirstEval = std::tuple_element_t<0, std::tuple<EvalTypes...>>;

public:
  using InputType = typename FirstEval::InputType;
  using OutputType = typename FirstEval::OutputType;
  static_assert((std::is_same_v<typename EvalTypes::InputType, InputType> && ...),
                "All FuncEval types must have the same InputType");
  static_assert((std::is_same_v<typename EvalTypes::OutputType, OutputType> && ...),
                "All FuncEval types must have the same OutputType");

  /// Construct from pre-built FuncEval objects
  C20CONSTEXPR explicit FuncEvalMany(EvalTypes... evals)
      : evals_{std::move(evals)...} {}

  /// Number of functions in the group
  [[nodiscard]] constexpr std::size_t size() const noexcept { return sizeof...(EvalTypes); }

  /// Return a tuple of each FuncEval::coeffs()
  constexpr auto coeffs() const { return coeffs_impl(std::make_index_sequence<sizeof...(EvalTypes)>{}); }

  // ---------------------------------------------------------------------------
  // Evaluation APIs
  // ---------------------------------------------------------------------------

  /// Evaluate all functions at a single input, returning tuple
  [[nodiscard]] constexpr auto operator()(InputType arg) const noexcept {
    return eval_impl(arg, std::make_index_sequence<sizeof...(EvalTypes)>{});
  }

  /// Evaluate with one argument per function (variadic), returning tuple
  template <typename... Args,
            std::enable_if_t<sizeof...(Args) == sizeof...(EvalTypes) &&
                             (std::is_convertible_v<Args, InputType> && ...), int> = 0>
  [[nodiscard]] constexpr auto operator()(Args&&... args) const noexcept {
    return eval_args_impl(std::make_index_sequence<sizeof...(EvalTypes)>{}, std::forward<Args>(args)...);
  }

  /// Evaluate with tuple of inputs (size N), returning tuple
  template <typename Tuple,
            std::enable_if_t<std::tuple_size_v<std::remove_reference_t<Tuple>> == sizeof...(EvalTypes), int> = 0>
  [[nodiscard]] constexpr auto operator()(Tuple&& tup) const noexcept {
    return std::apply([this](auto&&... elems) {
                        return (*this)(std::forward<decltype(elems)>(elems)...);
                      },
                      std::forward<Tuple>(tup));
  }

  /// Evaluate with array of inputs (size N), returning std::array
  [[nodiscard]] constexpr std::array<OutputType, sizeof...(EvalTypes)> operator()
      (const std::array<InputType, sizeof...(EvalTypes)>& inputs) const noexcept {
    return eval_array_impl(inputs, std::make_index_sequence<sizeof...(EvalTypes)>{});
  }

private:
  std::tuple<EvalTypes...> evals_;

  // Helper: evaluate all at same arg into tuple
  template <std::size_t... I>
  constexpr auto eval_impl(InputType arg, std::index_sequence<I...>) const noexcept {
    return std::make_tuple(std::get<I>(evals_)(arg)...);
  }

  // Helper: evaluate each with its own arg into tuple
  template <std::size_t... I, typename... Args>
  constexpr auto eval_args_impl(std::index_sequence<I...>, Args&&... args) const noexcept {
    auto packed = std::forward_as_tuple(std::forward<Args>(args)...);
    return std::make_tuple(std::get<I>(evals_)(std::get<I>(packed))...);
  }

  // Helper: evaluate array-input into std::array
  template <std::size_t... I>
  constexpr std::array<OutputType, sizeof...(EvalTypes)> eval_array_impl(
      const std::array<InputType, sizeof...(EvalTypes)>& inputs,
      std::index_sequence<I...>) const noexcept {
    return {{ std::get<I>(evals_)(inputs[I])... }};
  }

  // Helper: gather coeffs into tuple
  template <std::size_t... I>
  constexpr auto coeffs_impl(std::index_sequence<I...>) const {
    return std::make_tuple(std::get<I>(evals_).coeffs()...);
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

template <typename... EvalTypes>
C20CONSTEXPR FuncEvalMany<EvalTypes...> make_func_eval(EvalTypes... evals) noexcept;

} // namespace poly_eval

// Include implementations
// ReSharper disable once CppUnusedIncludeDirective
#include "fast_eval_impl.hpp"