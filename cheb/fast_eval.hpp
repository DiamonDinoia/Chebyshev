#pragma once

#include <array>
#include <cmath>
#include <experimental/mdspan>
#include <functional>
#include <type_traits>
#include <vector>

#include "macros.h"
#include "poly_eval.h"

namespace poly_eval {

template <typename T> struct function_traits;

template <typename T, typename> struct is_tuple_like;

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

  C20CONSTEXPR static Buffer<OutputType, N_compile_time> bjorck_pereyra(const Buffer<InputType, N_compile_time> &x,
                                                                        const Buffer<OutputType, N_compile_time> &y);

  C20CONSTEXPR static Buffer<OutputType, N_compile_time>
  newton_to_monomial(const Buffer<OutputType, N_compile_time> &alpha, const Buffer<InputType, N_compile_time> &nodes);

  C20CONSTEXPR void refine(const Buffer<InputType, N_compile_time> &x_cheb_,
                           const Buffer<OutputType, N_compile_time> &y_cheb_);

  // Friend declaration for FuncEvalMany to access private members
  template <typename... EvalTypes> friend class FuncEvalMany;
};

//======================================================================
//  FuncEvalMany  –  vector-friendly wrapper over several FuncEval’s
//======================================================================
template <typename... EvalTypes> class FuncEvalMany {
  static_assert(sizeof...(EvalTypes) > 0, "At least one FuncEval type is required");

  using FirstEval = std::tuple_element_t<0, std::tuple<EvalTypes...>>;
  using InputType = typename FirstEval::InputType;
  using OutputType = typename FirstEval::OutputType;

  static constexpr std::size_t kF = sizeof...(EvalTypes);
  static constexpr std::size_t deg_max_ctime_ = std::max({EvalTypes::kDegreeCompileTime...});

  std::size_t deg_max_ = deg_max_ctime_; // run-time degree

  // ----- column-major coefficient matrix  (rows = degree, cols = poly) -----
  static constexpr std::size_t dyn = std::experimental::dynamic_extent;
  using Ext = std::experimental::extents<std::size_t, (deg_max_ctime_ ? deg_max_ctime_ : dyn), kF>;

  Buffer<OutputType, kF * deg_max_ctime_> coeff_store_;
  std::experimental::mdspan<OutputType, Ext, std::experimental::layout_right> coeffs_{nullptr, 1, kF};

  // per-poly scaling
  std::array<InputType, kF> low_;
  std::array<InputType, kF> hi_;

public:
  // ------------------------------------------------------------------ ctor
  explicit FuncEvalMany(const EvalTypes &...evals) : low_{evals.low...}, hi_{evals.hi...} {
    if constexpr (deg_max_ctime_ == 0) {
      deg_max_ = std::max({evals.n_terms...});
      coeff_store_.assign(kF * deg_max_, OutputType{});
      coeffs_ = decltype(coeffs_){coeff_store_.data(), deg_max_, kF};
    } else {
      coeffs_ = decltype(coeffs_){coeff_store_.data(), deg_max_ctime_, kF};
    }
    copy_coeffs<0>(evals...);
  }

  [[nodiscard]] std::size_t size() const noexcept { return kF; }
  [[nodiscard]] std::size_t degree() const noexcept { return deg_max_; }

  // ------------------------------------------------ broadcast : one x
  [[nodiscard]] std::array<OutputType, kF> operator()(InputType x) const noexcept {
    std::array<InputType, kF> xu;
    for (std::size_t i = 0; i < kF; ++i)
      xu[i] = xsimd::fms(InputType(2.0), x, hi_[i]) * low_[i];

    std::array<OutputType, kF> res{};
    horner_transposed<kF, deg_max_ctime_>(xu.data(), coeffs_.data_handle(), res.data(), kF, deg_max_);
    return res;
  }

  // ------------------------------------------------ per-poly input array
  [[nodiscard]] std::array<OutputType, kF> operator()(const std::array<InputType, kF> &xs) const noexcept {
    std::array<InputType, kF> xu;
    for (std::size_t i = 0; i < kF; ++i)
      xu[i] = xsimd::fms(InputType(2.0), xs[i], hi_[i]) * low_[i];

    std::array<OutputType, kF> res{};
    horner_transposed<kF, deg_max_ctime_>(xu.data(), coeffs_.data_handle(), res.data(), kF, deg_max_);
    return res;
  }

  // ------------------------------------------------ variadic convenience
  template <typename... Ts>
  [[nodiscard]] std::array<OutputType, kF> operator()(InputType first, Ts... rest) const noexcept {
    static_assert(sizeof...(Ts) + 1 == kF, "Argument count must equal number of polynomials");
    return operator()(std::array<InputType, kF>{first, static_cast<InputType>(rest)...});
  }

  // ------------------------------------------------ tuple of inputs
  template <typename... Ts>
  [[nodiscard]] std::array<OutputType, kF> operator()(const std::tuple<Ts...> &tup) const noexcept {
    static_assert(sizeof...(Ts) == kF, "Tuple size must equal number of polynomials");
    std::array<InputType, kF> xs{};
    std::apply([&](auto &&...e) { xs = {static_cast<InputType>(e)...}; }, tup);
    return operator()(xs);
  }

private:
  // ---------- copy reversed coefficients into column-major matrix -----
  template <std::size_t I, typename FE, typename... Rest> void copy_coeffs(const FE &fe, const Rest &...rest) {
    for (std::size_t k = 0; k < fe.n_terms; ++k)
      coeffs_(k, I) = fe.monomials[k];
    for (std::size_t k = fe.n_terms; k < deg_max_; ++k)
      coeffs_(k, I) = OutputType{};
    if constexpr (I + 1 < kF)
      copy_coeffs<I + 1>(rest...);
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