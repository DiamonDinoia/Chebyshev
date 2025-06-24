#pragma once

#include "macros.h"
#include "utils.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <xsimd/xsimd.hpp>

namespace poly_eval {
template <std::size_t N_total = 0, typename OutputType, typename InputType>
ALWAYS_INLINE constexpr OutputType horner(const InputType x, const OutputType *c_ptr, std::size_t c_size = 0) noexcept {
  if constexpr (N_total != 0) {
    // Compile-time unrolled Horner
    // Start with highest-degree term
    OutputType acc = c_ptr[N_total - 1];
    // Unroll remaining N_total-1 steps
    detail::unroll_loop<N_total - 1>([&](auto idx_c) {
      const std::size_t i = idx_c; // value of the loop index
      const std::size_t k = (N_total - 2) - i;
      if constexpr (std::is_floating_point_v<OutputType>) {
        acc = detail::fma(acc, x, c_ptr[k]);
      } else {
        using std::real;
        using std::imag;
        acc = OutputType{detail::fma(real(acc), x, real(c_ptr[k])), detail::fma(imag(acc), x, imag(c_ptr[k]))};
      }
    });
    return acc;
  } else {
    // Runtime iterative Horner
    OutputType acc = c_ptr[c_size - 1];
    for (std::size_t k = c_size - 1; k-- > 0;) {
      if constexpr (std::is_floating_point_v<OutputType>) {
        acc = detail::fma(acc, x, c_ptr[k]);
      } else {
        using std::imag;
        using std::real;
        acc = OutputType{detail::fma(real(acc), x, real(c_ptr[k])), detail::fma(imag(acc), x, imag(c_ptr[k]))};
      }
    }
    return acc;
  }
}

//------------------------------------------------------------------------------
// SIMD Horner with optional unroll of coefficients
//------------------------------------------------------------------------------
/// @tparam OuterUnrollFactor Number of SIMD batches per iteration
/// @tparam pts_aligned       Are input points aligned
/// @tparam out_aligned       Are outputs aligned
/// @tparam N_monomials       Compile-time number of coefficients (0 = runtime)
/// @tparam InputType         Scalar input type
/// @tparam OutputType        Scalar output type
/// @tparam MapFunc           Function to map each input before evaluation
/// @param  pts               Input points array
/// @param  out               Output values array
/// @param  num_points        Number of points
/// @param  monomials         Coefficients array
/// @param  monomials_size    Number of coefficients (used if N_monomials==0)
/// @param  map_func          Optional mapping function (default identity)
template <std::size_t N_monomials = 0, bool pts_aligned = false, bool out_aligned = false, int UNROLL = 0,
          typename InputType, typename OutputType, typename MapFunc = decltype([](auto v) { return v; })>
ALWAYS_INLINE void horner(const InputType *pts, OutputType *out, std::size_t num_points, const OutputType *monomials,
                          std::size_t monomials_size, const MapFunc map_func = {}) noexcept {
  constexpr auto simd_size = xsimd::batch<InputType>::size;
  constexpr auto OuterUnrollFactor = UNROLL > 0 ? UNROLL : 32 / sizeof(OutputType);
  const std::size_t block = simd_size * OuterUnrollFactor;
  const std::size_t trunc = num_points & (-block);
  static constexpr auto pts_mode = [] {
    if constexpr (pts_aligned) {
      return xsimd::aligned_mode{};
    } else {
      return xsimd::unaligned_mode{};
    }
  }();
  static constexpr auto out_mode = [] {
    if constexpr (out_aligned) {
      return xsimd::aligned_mode{};
    } else {
      return xsimd::unaligned_mode{};
    }
  }();
  for (std::size_t i = 0; i < trunc; i += block) {
    xsimd::batch<InputType> pt_batches[OuterUnrollFactor];
    xsimd::batch<OutputType> acc_batches[OuterUnrollFactor];
    // Load and init
    detail::unroll_loop<OuterUnrollFactor>([&](auto j) {
      pt_batches[j] = map_func(xsimd::load(pts + i + j * simd_size, pts_mode));
      acc_batches[j] =
          xsimd::batch<OutputType>(N_monomials ? monomials[N_monomials - 1] : monomials[monomials_size - 1]);
    });

    // Horner steps: either compile-time unrolled or runtime loop
    if constexpr (N_monomials != 0) {
      // Unroll coefficient steps (descending)
      detail::unroll_loop<N_monomials - 1>([&](auto idx) {
        const std::size_t k = (N_monomials - 2) - idx;
        detail::unroll_loop<OuterUnrollFactor>([&](auto j) {
          if constexpr (std::is_floating_point_v<OutputType>) {
            acc_batches[j] = xsimd::fma(pt_batches[j], acc_batches[j], xsimd::batch<OutputType>(monomials[k]));
          } else {
            auto coeffs = xsimd::batch<OutputType>(monomials[k]);
            acc_batches[j] = {xsimd::fma(real(acc_batches[j]), pt_batches[j], real(coeffs)),
                              xsimd::fma(imag(acc_batches[j]), pt_batches[j], imag(coeffs))};
          }
        });
      });
    } else {
      // Runtime coefficient loop
      for (std::size_t k = monomials_size - 1; k-- > 0;) {
        detail::unroll_loop<OuterUnrollFactor>([&](auto j) {
          if constexpr (std::is_floating_point_v<OutputType>) {
            acc_batches[j] = xsimd::fma(pt_batches[j], acc_batches[j], xsimd::batch<OutputType>(monomials[k]));
          } else {
            auto coeffs = xsimd::batch<OutputType>(monomials[k]);
            acc_batches[j] = {xsimd::fma(real(acc_batches[j]), pt_batches[j], real(coeffs)),
                              xsimd::fma(imag(acc_batches[j]), pt_batches[j], imag(coeffs))};
          }
        });
      }
    }

    // Store
    detail::unroll_loop<OuterUnrollFactor>([&](auto j) { acc_batches[j].store(out + i + j * simd_size, out_mode); });
  }

  // Remainder
  for (std::size_t i = trunc; i < num_points; ++i) {
    out[i] = horner<N_monomials>(map_func(pts[i]), monomials, N_monomials ? N_monomials : monomials_size);
  }
}

} // namespace poly_eval