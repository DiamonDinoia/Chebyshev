#pragma once
#if __cplusplus >= 202002L // Check for C++20 or later
#include <bit>
#endif
#include <cstdint>
#include "macros.h"


namespace poly_eval::detail {
template <typename T>
ALWAYS_INLINE static constexpr size_t get_alignment(const T *ptr) noexcept {
  const auto address = reinterpret_cast<uintptr_t>(ptr);

  if (address == 0) {
    // A null pointer (or an address of 0) doesn't have a meaningful alignment
    // in the context of data access. You could return 0 or 1 depending on
    // your specific interpretation, but 0 makes sense here.
    return 0;
  }
  // Modern C++ (C++20 and later) has built-in functions for this
#if __cplusplus >= 202002L // Check for C++20 or later
  // std::countr_zero returns the number of trailing zero bits.
  // If an address is N-byte aligned, its N lowest bits must be zero.
  // So, if an address is 8-byte aligned (e.g., 0x...1000), it has 3 trailing zeros.
  // 2^3 = 8.
  return static_cast<size_t>(1) << std::countr_zero(address);
#else
  // For older C++ versions, we can use a loop or compiler intrinsics.
  // This loop finds the least significant '1' bit, which determines the alignment.
  size_t alignment = 1;
  while ((address & alignment) == 0 && (alignment < (static_cast<size_t>(1) << (sizeof(uintptr_t) * 8 - 1)))) {
    alignment <<= 1; // Multiply by 2
  }
  return alignment;
#endif
}

template <typename F, std::size_t... Is>
ALWAYS_INLINE constexpr void unroll_loop_impl(F &&func, std::index_sequence<Is...>) {
  (func(Is), ...); // C++17 fold expression for comma operator
}

template <std::size_t Count, typename F>
ALWAYS_INLINE constexpr void unroll_loop(F &&func) {
  unroll_loop_impl(std::forward<F>(func), std::make_index_sequence<Count>{});
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
    double r = std::fma(c6, z, c5);
    r = std::fma(r, z, c4);
    r = std::fma(r, z, c3);
    r = std::fma(r, z, c2);
    r = std::fma(r, z, c1);
    return std::fma(z * z, r, 1.0 - 0.5 * z);
  };

  constexpr auto sin_poly = [](const double yy) constexpr {
    constexpr double s1 = -1.66666666666666307295e-01;
    constexpr double s2 = 8.33333333332211858878e-03;
    constexpr double s3 = -1.98412698295895385996e-04;
    constexpr double s4 = 2.75573136213857245213e-06;
    constexpr double s5 = -2.50507477628578072866e-08;
    constexpr double s6 = 1.58962301576546568060e-10;
    const double z = yy * yy;
    double r = std::fma(s6, z, s5);
    r = std::fma(r, z, s4);
    r = std::fma(r, z, s3);
    r = std::fma(r, z, s2);
    r = std::fma(r, z, s1);
    return std::fma(yy * z, r, yy);
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

}