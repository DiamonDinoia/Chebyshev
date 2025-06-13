#pragma once

#include "macros.h"
#include "utils.h"

#include <cassert>
#include <iomanip>
#include <iostream>
#include <xsimd/xsimd.hpp>

namespace poly_eval {

namespace detail {
// Runtime version (for compatibility with std::vector based linspace in other APIs)
template <typename T> std::vector<T> linspace(T start, T end, int num_points) {
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
} // namespace detail

// -----------------------------------------------------------------------------
// FuncEval Implementation (Runtime)
// -----------------------------------------------------------------------------

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
template <std::size_t CurrentN, typename>
C20CONSTEXPR FuncEval<Func, N_compile_time, Iters_compile_time>::FuncEval(Func F, const int n, const InputType a,
                                                                          const InputType b, const InputType *pts)
    : n_terms(n), low(b - a), hi(b + a) {
  assert(n_terms > 0 && "Polynomial degree must be positive");
  monomials.resize(n_terms);
  initialize_monomials(F, pts);
}

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
template <std::size_t CurrentN, typename>
C20CONSTEXPR FuncEval<Func, N_compile_time, Iters_compile_time>::FuncEval(Func F, const InputType a, const InputType b,
                                                                          const InputType *pts)
    : n_terms(static_cast<int>(CurrentN)), low(b - a), hi(b + a) {
  assert(n_terms > 0 && "Polynomial degree must be positive (template N > 0)");
  initialize_monomials(F, pts);
}

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
FAST_MATH_BEGIN typename FuncEval<Func, N_compile_time, Iters_compile_time>::OutputType C20CONSTEXPR
FuncEval<Func, N_compile_time, Iters_compile_time>::operator()(const InputType pt) const noexcept {
  const auto xi = map_from_domain(pt);
  return horner(monomials.data(), monomials.size(), xi); // Pass data pointer and size
}

FAST_MATH_END

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
template <int K_Current, int K_Target, int OuterUnrollFactor, class VecInputType, class VecOutputType>
FAST_MATH_BEGIN ALWAYS_INLINE void
FuncEval<Func, N_compile_time, Iters_compile_time>::horner(VecInputType *RESTRICT pt_batches,
                                                           VecOutputType *RESTRICT acc_batches) const noexcept {
  // Array of accumulator batches
  // No monomials_ptr needed; 'this->monomials' is accessible

  // Base case for the recursion: if K_Current is less than K_Target, stop.
  if constexpr (K_Current >= K_Target) {
    // Inner loop unrolling for 'j' (OuterUnrollFactor)
    // This uses a C++17 generic lambda with a fold expression to unroll the inner loop
    // for each batch within the current unroll factor.
    [&]<std::size_t... J>(std::integer_sequence<std::size_t, J...>) {
      ((
           // Apply Horner's method: acc = pt * acc + monomials[K_Current]
           // monomials is now accessed implicitly via the 'this' pointer
           acc_batches[J] = xsimd::fma(pt_batches[J], acc_batches[J], xsimd::batch<OutputType>(monomials[K_Current]))),
       ...); // Fold expression applies the operation for each J in the sequence
    }(std::make_integer_sequence<std::size_t, OuterUnrollFactor>{});

    // Recursive call to process the next coefficient (k-1)
    horner<K_Current - 1, K_Target, OuterUnrollFactor>(pt_batches, acc_batches);
  }
}

FAST_MATH_END
// Batch evaluation implementation using SIMD and unrolling
template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
template <int OuterUnrollFactor, bool pts_aligned, bool out_aligned>
FAST_MATH_BEGIN ALWAYS_INLINE void FuncEval<Func, N_compile_time, Iters_compile_time>::horner_polyeval(
    const InputType *RESTRICT pts, OutputType *RESTRICT out, std::size_t num_points) const noexcept {

  static_assert(OuterUnrollFactor > 0 && (OuterUnrollFactor & (OuterUnrollFactor - 1)) == 0,
                "OuterUnrollFactor must be a power of two greater than zero.");
  static constexpr auto simd_size = xsimd::batch<InputType>::size;
  const auto monomials_ptr = monomials.data();
  const auto monomials_size = monomials.size();
  const auto trunc_size = num_points & (-simd_size * OuterUnrollFactor);

  static constexpr auto pts_aligment = [] {
    if constexpr (pts_aligned) {
      return xsimd::aligned_mode{};
    } else {
      return xsimd::unaligned_mode{};
    }
  }();
  static constexpr auto out_aligment = [] {
    if constexpr (out_aligned) {
      return xsimd::aligned_mode{};
    } else {
      return xsimd::unaligned_mode{};
    }
  }();

  for (std::size_t i = 0; i < trunc_size; i += simd_size * OuterUnrollFactor) {
    // Use arrays to hold the batches and accumulators
    xsimd::batch<InputType> pt_batches[OuterUnrollFactor];
    xsimd::batch<OutputType> acc_batches[OuterUnrollFactor];

    // Load input points and initialize accumulators in a loop
    detail::unroll_loop<OuterUnrollFactor>([&](const auto j) {
      // Ensure we don't read past num_points for partial blocks
      pt_batches[j] = map_from_domain(xsimd::load(pts + i + j * simd_size, pts_aligment));
      // Initialize with the last coefficient
      acc_batches[j] = xsimd::batch<OutputType>(monomials_ptr[monomials_size - 1]);
    });

    // Process each batch in the inner loop (Horner's method)
    // Iterating from monomials_size - 2 down to 0
    if constexpr (N_compile_time > 0) {
      horner<static_cast<int>(N_compile_time - 1), 0, OuterUnrollFactor>(pt_batches, acc_batches);
    } else {
      for (int k = monomials_size - 2; k >= 0; --k) {
        detail::unroll_loop<OuterUnrollFactor>([&](const auto j) {
          // acc_batches[j] = pt_batches[j] * acc_batches[j] + monomials_ptr[k]
          acc_batches[j] = xsimd::fma(pt_batches[j], acc_batches[j], xsimd::batch<OutputType>(monomials_ptr[k]));
        });
      }
    }

    // Store results in a loop
    detail::unroll_loop<OuterUnrollFactor>(
        [&](const auto j) { xsimd::store(out + i + j * simd_size, acc_batches[j], out_aligment); });
  }
  // Handle any remaining points that didn't fit into the full unrolled blocks
  for (int i = trunc_size; i < num_points; ++i) {
    out[i] = operator()(pts[i]);
  }
}

FAST_MATH_END

// Batch evaluation implementation using SIMD and unrolling
template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
template <int OuterUnrollFactor, bool pts_aligned, bool out_aligned>
FAST_MATH_BEGIN NO_INLINE void FuncEval<Func, N_compile_time, Iters_compile_time>::no_inline_horner_polyeval(
    const InputType *RESTRICT pts, OutputType *RESTRICT out, std::size_t num_points) const noexcept {
  return horner_polyeval<OuterUnrollFactor, pts_aligned, out_aligned>(pts, out, num_points);
}

FAST_MATH_END

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
template <bool pts_aligned, bool out_aligned>
FAST_MATH_BEGIN void
FuncEval<Func, N_compile_time, Iters_compile_time>::operator()(const InputType *RESTRICT pts, OutputType *RESTRICT out,
                                                               std::size_t num_points) const noexcept {
  // find out the alignment of pts and out
  constexpr auto simd_size = xsimd::batch<InputType>::size;
  constexpr auto alignment = xsimd::best_arch::alignment();
  constexpr auto unroll_factor = 4;

  const auto monomial_ptr = monomials.data();
  const auto monomial_size = monomials.size();

  if constexpr (pts_aligned) {
    if constexpr (out_aligned) {
      return horner_polyeval<unroll_factor, true, true>(pts, out, num_points);
    } else {
      return horner_polyeval<unroll_factor, true, false>(pts, out, num_points);
    }
  } else {
    const auto pts_alignment = detail::get_alignment(pts);
    const auto out_alignment = detail::get_alignment(out);
    if (pts_alignment != out_alignment) {
      if (pts_alignment >= alignment) {
        return no_inline_horner_polyeval<unroll_factor, true, false>(pts, out, num_points);
      }
      if (out_alignment >= alignment) {
        return no_inline_horner_polyeval<unroll_factor, false, true>(pts, out, num_points);
      }
      return no_inline_horner_polyeval<unroll_factor, false, false>(pts, out, num_points);
    }

    // process scalar until we reach the first aligned point
    std::size_t i;
    for (i = 0; i < num_points % alignment; ++i) {
      out[i] = operator()(pts[i]);
    }
    return horner_polyeval<unroll_factor, true, true>(pts + i, out + i, num_points - i);
  }
}

FAST_MATH_END

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
C20CONSTEXPR const Buffer<typename FuncEval<Func, N_compile_time, Iters_compile_time>::OutputType, N_compile_time> &
FuncEval<Func, N_compile_time, Iters_compile_time>::coeffs() const noexcept {
  return monomials;
}

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
C20CONSTEXPR void FuncEval<Func, N_compile_time, Iters_compile_time>::initialize_monomials(Func F,
                                                                                           const InputType *pts) {
  std::vector<InputType> grid{};
  std::vector<OutputType> samples{};
  grid.resize(n_terms);
  for (int k = 0; k < n_terms; ++k) {
    grid[k] = pts == nullptr ? static_cast<InputType>(detail::cos((2.0 * k + 1.0) * M_PI / (2.0 * n_terms))) : pts[k];
  }
  samples.resize(n_terms);
  for (int i = 0; i < n_terms; ++i) {
    samples[i] = F(map_to_domain(grid[i]));
  }
  const auto newton = bjorck_pereyra(grid, samples);
  const auto temp_monomials = newton_to_monomial(newton, grid);
  assert(temp_monomials.size() == monomials.size() && "Monomial coefficients size mismatch after conversion!");
  std::copy(temp_monomials.begin(), temp_monomials.end(), monomials.begin());
  refine(grid, samples);
}

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
template <class T>
FAST_MATH_BEGIN constexpr T FuncEval<Func, N_compile_time, Iters_compile_time>::map_to_domain(const T T_arg) const {
  return static_cast<T>(0.5 * (low * T_arg + hi));
}

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
template <class T>
constexpr T FuncEval<Func, N_compile_time, Iters_compile_time>::map_from_domain(const T T_arg) const {
  return static_cast<T>((2.0 * T_arg - hi) / low);
}

FAST_MATH_END

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
template <std::size_t N_total, std::size_t current_idx>
typename FuncEval<Func, N_compile_time, Iters_compile_time>::OutputType
    FAST_MATH_BEGIN constexpr FuncEval<Func, N_compile_time, Iters_compile_time>::horner(
        const OutputType *RESTRICT c_ptr, InputType x) noexcept {
  if constexpr (current_idx == N_total - 1) {
    return c_ptr[current_idx];
  } else {
    if constexpr (std::is_same_v<InputType, OutputType> && std::is_floating_point_v<InputType>) {
      return std::fma(horner<N_total, current_idx + 1>(c_ptr, x), x, c_ptr[current_idx]);
    } else {
      return horner<N_total, current_idx + 1>(c_ptr, x) * x + c_ptr[current_idx];
    }
  }
}

FAST_MATH_END

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
typename FuncEval<Func, N_compile_time, Iters_compile_time>::OutputType
    FAST_MATH_BEGIN constexpr FuncEval<Func, N_compile_time, Iters_compile_time>::horner(
        const OutputType *RESTRICT c_ptr, std::size_t c_size, InputType x) noexcept {
  if constexpr (N_compile_time != 0) {
    return horner<N_compile_time, 0>(c_ptr, x); // Use compile-time N
  } else {
    OutputType acc = c_ptr[c_size - 1]; // Start with the highest degree coefficient
    for (int k = static_cast<int>(c_size) - 2; k >= 0; --k) {
      if constexpr (std::is_same_v<InputType, OutputType> && std::is_floating_point_v<InputType>) {
        acc = xsimd::fma(x, acc, c_ptr[k]);
      } else {
        acc = acc * x + c_ptr[k];
      }
    }
    return acc;
  }
}

FAST_MATH_END

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
std::vector<typename FuncEval<Func, N_compile_time, Iters_compile_time>::OutputType>
    C20CONSTEXPR FuncEval<Func, N_compile_time, Iters_compile_time>::bjorck_pereyra(const std::vector<InputType> &x,
                                                                                    const std::vector<OutputType> &y) {
  const auto n = static_cast<int>(x.size());
  std::vector<OutputType> a = y;
  for (std::size_t k = 0; k < n - 1; ++k) {
    for (std::size_t i = n - 1; i >= k + 1; --i) {
      a[i] = (a[i] - a[i - 1]) / static_cast<OutputType>(x[i] - x[i - k - 1]);
    }
  }
  return a;
}

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
std::vector<typename FuncEval<Func, N_compile_time, Iters_compile_time>::OutputType> C20CONSTEXPR
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
C20CONSTEXPR void FuncEval<Func, N_compile_time, Iters_compile_time>::refine(const std::vector<InputType> &x_cheb_,
                                                                             const std::vector<OutputType> &y_cheb_) {
  for (std::size_t pass = 0; pass < kItersCompileTime; ++pass) {
    std::vector<OutputType> r_cheb(n_terms);
    for (int i = 0; i < n_terms; ++i) {
      InputType xi = x_cheb_[i];
      OutputType p_val = horner(this->monomials.data(), this->monomials.size(), xi); // Pass data pointer and size
      r_cheb[i] = y_cheb_[i] - p_val;
    }
    std::vector<OutputType> newton_r = bjorck_pereyra(x_cheb_, r_cheb);
    std::vector<OutputType> mono_r = newton_to_monomial(newton_r, x_cheb_);
    assert(mono_r.size() == monomials.size() && "Refinement coefficients size mismatch!");
    for (int j = 0; j < n_terms; ++j) {
      monomials[j] += mono_r[j];
    }
  }
}



// -----------------------------------------------------------------------------
// make_func_eval API implementations (Runtime, C++17 compatible)
// -----------------------------------------------------------------------------

template <std::size_t N_compile_time, std::size_t Iters_compile_time, class Func>
NO_INLINE C20CONSTEXPR auto make_func_eval(Func F, typename function_traits<Func>::arg0_type a,
                                           typename function_traits<Func>::arg0_type b,
                                           const typename function_traits<Func>::arg0_type *pts) {
  return FuncEval<Func, N_compile_time, Iters_compile_time>(F, a, b, pts);
}

template <std::size_t Iters_compile_time, class Func>
NO_INLINE C20CONSTEXPR auto make_func_eval(Func F, int n, typename function_traits<Func>::arg0_type a,
                                           typename function_traits<Func>::arg0_type b,
                                           const typename function_traits<Func>::arg0_type *pts) {
  return FuncEval<Func, 0, Iters_compile_time>(F, n, a, b, pts);
}

// C++17 compatible API for finding minimum N for a given error tolerance (runtime eps)
template <std::size_t MaxN_val, std::size_t NumEvalPoints_val, std::size_t Iters_compile_time, class Func>
NO_INLINE C20CONSTEXPR auto make_func_eval(Func F, double eps, // eps as a runtime parameter
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
        std::cerr << "Warning: Requested epsilon " << eps << " is less than machine epsilon for OutputType ("
                  << std::numeric_limits<OutputType>::epsilon() << "). Clamping.\n";
        eps = std::numeric_limits<OutputType>::epsilon();
      }
    } else if constexpr (std::is_same_v<OutputType, std::complex<float>>) {
      if (eps < std::numeric_limits<float>::epsilon()) {
        std::cerr << "Warning: Requested epsilon " << eps << " is less than machine epsilon for std::complex<float> ("
                  << std::numeric_limits<float>::epsilon() << "). Clamping.\n";
        eps = std::numeric_limits<float>::epsilon();
      }
    } else if constexpr (std::is_same_v<OutputType, std::complex<double>>) {
      if (eps < std::numeric_limits<double>::epsilon()) {
        std::cerr << "Warning: Requested epsilon " << eps << " is less than machine epsilon for std::complex<double> ("
                  << std::numeric_limits<double>::epsilon() << "). Clamping.\n";
        eps = std::numeric_limits<double>::epsilon();
      }
    }
  }

  std::vector<InputType> eval_points = detail::linspace(a, b, static_cast<int>(NumEvalPoints_val));

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
      std::cout << "Converged: Found min degree N = " << n << " (Max Error: " << std::scientific << std::setprecision(4)
                << max_observed_error << " <= Epsilon: " << eps << ")\n";
      return current_evaluator;
    }
  }
  std::cout << "Warning: Did not converge to epsilon " << std::scientific << std::setprecision(4) << eps
            << " within MaxN = " << MaxN_val << ". Returning FuncEval with degree " << MaxN_val << ".\n";
  return FuncEval<Func, 0, Iters_compile_time>(F, static_cast<int>(MaxN_val), a, b);
}

#if __cplusplus >= 202002L
template <double eps_val, std::size_t MaxN_val, std::size_t NumEvalPoints_val, std::size_t Iters_compile_time,
          class Func>
NO_INLINE constexpr auto make_func_eval(Func F, typename function_traits<Func>::arg0_type a,
                                        typename function_traits<Func>::arg0_type b) {
  using InputType = typename function_traits<Func>::arg0_type;
  using OutputType = typename function_traits<Func>::result_type;

  static_assert(MaxN_val > 0, "Max polynomial degree must be positive.");
  static_assert(NumEvalPoints_val > 1, "Number of evaluation points must be greater than 1.");

  std::vector<InputType> eval_points = detail::linspace(a, b, static_cast<int>(NumEvalPoints_val));

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
      std::cout << "Converged: Found min degree N = " << n << " (Max Error: " << std::scientific << std::setprecision(4)
                << max_observed_error << " <= Epsilon: " << eps_val << ")\n";
      return current_evaluator;
    }
  }

  std::cout << "Warning: Did not converge to epsilon " << std::scientific << std::setprecision(4) << eps_val
            << " within MaxN = " << MaxN_val << ". Returning FuncEval with degree " << MaxN_val << ".\n";
  return FuncEval<Func, 0, Iters_compile_time>(F, static_cast<int>(MaxN_val), a, b);
}
#endif


template <typename... EvalTypes>
C20CONSTEXPR FuncEvalMany<EvalTypes...> make_func_eval_many(EvalTypes... evals) noexcept {
  return FuncEvalMany<EvalTypes...>(std::move(evals)...);
}


} // namespace poly_eval