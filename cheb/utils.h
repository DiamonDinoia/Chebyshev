#pragma once
#if defined(__cpp_lib_bitops) && (__cpp_lib_bitops >= 201907L)
#include <bit>
#endif
#include <cmath>
#include <cstddef>
#include <functional>
#include <utility>
#include <xsimd/xsimd.hpp>

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
// -----------------------------------------------------------------------------
// function_traits: Helper to deduce input and output types from a callable
// -----------------------------------------------------------------------------
template <typename T> struct function_traits : function_traits<decltype(&T::operator())> {};

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
struct tuple_size_or_zero<T, std::void_t<decltype(std::tuple_size_v<std::remove_cvref_t<T>>)>>
    : std::integral_constant<std::size_t, std::tuple_size_v<std::remove_cvref_t<T>>> {};

} // namespace poly_eval

namespace poly_eval::detail {

template <typename T> constexpr ALWAYS_INLINE T fma(const T &a, const T &b, const T &c) noexcept {
  // Fused multiply-add: a * b + c
  if constexpr (std::is_floating_point_v<T>) {
    return std::fma(a, b, c);
  }
  return xsimd::fma(a, b, c);
}

template <typename U>
constexpr ALWAYS_INLINE auto fma(const std::complex<U> &a, const U &b, const std::complex<U> &c) noexcept {
  //  a * b + c  with fused ops on each component
  return std::complex<U>{xsimd::fma(real(a), b, real(c)), xsimd::fma(imag(a), b, imag(c))};
}

template <typename U>
constexpr ALWAYS_INLINE auto fma(const xsimd::batch<std::complex<U>> &a, const xsimd::batch<U> &b,
                                 const xsimd::batch<std::complex<U>> &c) noexcept {
  //  a * b + c  with fused ops on each component
  return xsimd::batch<std::complex<U>>{xsimd::fma(real(a), b, real(c)), xsimd::fma(imag(a), b, imag(c))};
}

// std::countr_zero returns the number of trailing zero bits.
// If an address is N-byte aligned, its N lowest bits must be zero.
// So, if an address is 8-byte aligned (e.g., 0x...1000), it has 3 trailing zeros.
// 2^3 = 8.
template <typename T> constexpr auto countr_zero(T x) noexcept {
  static_assert(std::is_unsigned_v<T>, "cntz requires an unsigned integral type");
  static_assert(std::is_unsigned<T>::value, "countr_zero_impl requires an unsigned type");
#if defined(__cpp_lib_bitops) && (__cpp_lib_bitops >= 201907L)
  // C++20: hand work to the standard library
  return std::countr_zero(x);
#else
  // constexpr portable fallback
  constexpr int w = std::numeric_limits<T>::digits;
  if (x == 0)
    return w;
  int n = 0;
  while ((x & T{1}) == T{0}) {
    x >>= 1;
    ++n;
  }
  return n;
#endif
}

template <typename T> constexpr size_t get_alignment(const T *ptr) noexcept {
  const auto address = reinterpret_cast<uintptr_t>(ptr);
  if (address == 0) {
    // A null pointer (or an address of 0) doesn't have a meaningful alignment
    // in the context of data access.
    return 0;
  }
  return static_cast<size_t>(1) << detail::countr_zero(address);
}

// Runtime unroll implementation
template <std::size_t Start, std::size_t Inc, typename F, std::size_t... Is>
ALWAYS_INLINE constexpr void unroll_loop_impl_runtime(F &&func, std::index_sequence<Is...>) {
  (func(Start + Is * Inc), ...);
}

// Compile-time unroll implementation
template <std::size_t Start, std::size_t Inc, typename F, std::size_t... Is>
ALWAYS_INLINE constexpr void unroll_loop_impl_constexpr(F &&func, std::index_sequence<Is...>) {
  (func.template operator()<Start + Is * Inc>(), ...);
}

// Helper: compute number of steps
template <std::size_t Start, std::size_t Stop, std::size_t Inc>
inline constexpr std::size_t compute_range_count = (Start < Stop) ? ((Stop - Start + Inc - 1) / Inc) : 0;

// Trait: detect runtime index
template <typename F> using is_runtime_callable = std::is_invocable<F, std::size_t>;

// Primary interface
// Runtime version
template <std::size_t Stop, std::size_t Start = 0, std::size_t Inc = 1, typename F,
          std::enable_if_t<is_runtime_callable<F>::value, int> = 0>
ALWAYS_INLINE constexpr void unroll_loop(F &&func) {
  constexpr std::size_t Count = compute_range_count<Start, Stop, Inc>;
  unroll_loop_impl_runtime<Start, Inc>(std::forward<F>(func), std::make_index_sequence<Count>{});
}

// Compile-time version
template <std::size_t Stop, std::size_t Start = 0, std::size_t Inc = 1, typename F,
          std::enable_if_t<!is_runtime_callable<F>::value, int> = 0>
ALWAYS_INLINE constexpr void unroll_loop(F &&func) {
  constexpr std::size_t Count = compute_range_count<Start, Stop, Inc>;
  unroll_loop_impl_constexpr<Start, Inc>(std::forward<F>(func), std::make_index_sequence<Count>{});
}

template <class T, uint8_t N = 1> constexpr uint8_t min_simd_width() {
  // finds the smallest simd width that can handle N elements
  // simd size is batch size the SIMD width in xsimd terminology
  if constexpr (std::is_void_v<xsimd::make_sized_batch_t<T, N>>) {
    return min_simd_width<T, N * 2>();
  } else {
    return N;
  }
};

template <typename T> constexpr uint8_t best_simd(std::size_t N) {
  // must have at least 2-wide SIMD on this arch
  constexpr uint8_t max_w = xsimd::batch<T, xsimd::best_arch>::size;
  static_assert(max_w >= 2, "Need at least 2-wide SIMD for this type/arch");

  // start at the smallest vector width (at least 2)
  uint8_t min_w = std::max<uint8_t>(min_simd_width<T>(), 2);
  uint8_t chosen = min_w;

  // only consider widths up to N
  for (uint8_t w = min_w; w <= max_w; w <<= 1) {
    if (w > N)
      break;

    std::size_t groups = (N + w - 1) / w;
    std::size_t padding = groups * w - N;

    // accept any w that wastes ≤ w/2 lanes
    if (padding <= w / 2) {
      chosen = w;
    }
  }

  return chosen;
}

template <typename T, std::size_t N> uint8_t alignment() {
  return xsimd::make_sized_batch_t<T, N>::arch_type::alignment();
}

constexpr double cos(const double x) noexcept {
  /* π/2 split (Cody-Waite) */
  constexpr double PIO2_HI = 1.57079632679489655800e+00;
  constexpr double PIO2_LO = 6.12323399573676603587e-17;
  constexpr double INV_PIO2 = 6.36619772367581382433e-01;

  if (!std::isfinite(x))
    return std::numeric_limits<double>::quiet_NaN();

  /* argument reduction: x = n·π/2 + y, |y| ≤ π/4 */

  const double fn = x * INV_PIO2;
  const int n = static_cast<int>(fn + (fn >= 0.0 ? 0.5 : -0.5));
  const int q = n & 3; // quadrant 0‥3
  const auto y = [n, x] {
    double y = std::fma(-n, PIO2_HI, x);
    y = std::fma(-n, PIO2_LO, y);
    return y;
  }();
  /* cos & sin minimax polynomials as lambdas with embedded coeffs */
  constexpr auto cos_poly = [](const double yy) constexpr {
    /*
     * The coefficients c1-c6 are under the following license:
     * ====================================================
     * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
     *
     * Developed at SunSoft, a Sun Microsystems, Inc. business.
     * Permission to use, copy, modify, and distribute this
     * software is freely granted, provided that this notice
     * is preserved.
     * ====================================================
     */
    constexpr double c1 = 4.16666666666666019037e-02;
    constexpr double c2 = -1.38888888888741095749e-03;
    constexpr double c3 = 2.48015872894767294178e-05;
    constexpr double c4 = -2.75573143513906633035e-07;
    constexpr double c5 = 2.08757232129817482790e-09;
    constexpr double c6 = -1.13596475577881948265e-11;
    const double z = yy * yy;
    double r = fma(c6, z, c5);
    r = fma(r, z, c4);
    r = fma(r, z, c3);
    r = fma(r, z, c2);
    r = fma(r, z, c1);
    return fma(z * z, r, 1.0 - 0.5 * z);
  };

  constexpr auto sin_poly = [](const double yy) constexpr {
    constexpr double s1 = -1.66666666666666307295e-01;
    constexpr double s2 = 8.33333333332211858878e-03;
    constexpr double s3 = -1.98412698295895385996e-04;
    constexpr double s4 = 2.75573136213857245213e-06;
    constexpr double s5 = -2.50507477628578072866e-08;
    constexpr double s6 = 1.58962301576546568060e-10;
    const double z = yy * yy;
    double r = fma(s6, z, s5);
    r = fma(r, z, s4);
    r = fma(r, z, s3);
    r = fma(r, z, s2);
    r = fma(r, z, s1);
    return fma(yy * z, r, yy);
  };

  /* quadrant dispatch—only compute what we need */
  switch (q) {
  case 0:
    return cos_poly(y);
  case 1:
    return -sin_poly(y);
  case 2:
    return -cos_poly(y);
  default:
    return sin_poly(y);
  }
}

} // namespace poly_eval::detail