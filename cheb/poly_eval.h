#pragma once

#include "macros.h"
#include "utils.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <xsimd/xsimd.hpp>

namespace poly_eval {

// Reversed Horner: coefficients array is ordered from highest-degree term at index 0
template <std::size_t N_total = 0, typename OutputType, typename InputType>
ALWAYS_INLINE constexpr OutputType horner(const InputType x, const OutputType *c_ptr, std::size_t c_size = 0) noexcept {
  if constexpr (N_total != 0) {
    // Compile-time unrolled Horner on reversed array
    // Start with highest-degree term at c_ptr[0]
    OutputType acc = c_ptr[0];
    // Unroll remaining N_total-1 steps: apply coefficients c_ptr[1] ... c_ptr[N_total-1]
    detail::unroll_loop<N_total - 1>([&](const auto idx) {
      const std::size_t k = idx + 1; // 1 ... N_total-1
      if constexpr (std::is_floating_point_v<OutputType>) {
        acc = detail::fma(acc, x, c_ptr[k]);
      } else {
        acc = OutputType{detail::fma(real(acc), x, real(c_ptr[k])), detail::fma(imag(acc), x, imag(c_ptr[k]))};
      }
    });
    return acc;
  } else {
    // Runtime iterative Horner on reversed array
    OutputType acc = c_ptr[0];
    for (std::size_t k = 1; k < c_size; ++k) {
      if constexpr (std::is_floating_point_v<OutputType>) {
        acc = detail::fma(acc, x, c_ptr[k]);
      } else {
        acc = OutputType{detail::fma(real(acc), x, real(c_ptr[k])), detail::fma(imag(acc), x, imag(c_ptr[k]))};
      }
    }
    return acc;
  }
}

//------------------------------------------------------------------------------
// SIMD Horner (coefficients reversed)
//------------------------------------------------------------------------------

template <std::size_t N_monomials = 0, bool pts_aligned = false, bool out_aligned = false, int UNROLL = 0,
          typename InputType, typename OutputType, typename MapFunc>
ALWAYS_INLINE constexpr void horner(const InputType *pts, OutputType *out, std::size_t num_points,
                                    const OutputType *monomials, std::size_t monomials_size,
                                    const MapFunc map_func = [](auto v) { return v; }) noexcept {
  C23STATIC constexpr auto simd_size = xsimd::batch<InputType>::size;
  C23STATIC constexpr auto OuterUnrollFactor = UNROLL > 0 ? UNROLL : 32 / sizeof(OutputType);
  C23STATIC constexpr auto block = simd_size * OuterUnrollFactor;
  C23STATIC constexpr auto pts_mode = [] {
    if constexpr (pts_aligned)
      return xsimd::aligned_mode{};
    else
      return xsimd::unaligned_mode{};
  }();
  C23STATIC constexpr auto out_mode = [] {
    if constexpr (out_aligned)
      return xsimd::aligned_mode{};
    else
      return xsimd::unaligned_mode{};
  }();

  const std::size_t trunc = num_points & (-block);


  for (std::size_t i = 0; i < trunc; i += block) {
    xsimd::batch<InputType> pt_batches[OuterUnrollFactor]{};
    xsimd::batch<OutputType> acc_batches[OuterUnrollFactor]{};

    // Load and initialize with highest-degree term monomials[0]
    detail::unroll_loop<OuterUnrollFactor>([&](const auto j) {
      pt_batches[j] = map_func(xsimd::load(pts + i + j * simd_size, pts_mode));
      acc_batches[j] = xsimd::batch<OutputType>(monomials[0]);
    });

    // Horner steps: ascending coefficients
    if constexpr (N_monomials != 0) {
      // Unroll compile-time
      detail::unroll_loop<N_monomials - 1>([&](const auto idx) {
        const std::size_t k = idx + 1; // 1 ... N_monomials-1
        detail::unroll_loop<OuterUnrollFactor>([&](const auto &j) {
          if constexpr (std::is_floating_point_v<OutputType>) {
            acc_batches[j] = detail::fma(acc_batches[j], pt_batches[j], xsimd::batch<OutputType>(monomials[k]));
          } else {
            auto coeff = xsimd::batch<OutputType>(monomials[k]);
            acc_batches[j] = {detail::fma(real(acc_batches[j]), pt_batches[j], real(coeff)),
                              detail::fma(imag(acc_batches[j]), pt_batches[j], imag(coeff))};
          }
        });
      });
    } else {
      // Runtime loop over remaining coefficients
      for (std::size_t k = 1; k < monomials_size; ++k) {
        detail::unroll_loop<OuterUnrollFactor>([&](const auto j) {
          if constexpr (std::is_floating_point_v<OutputType>) {
            acc_batches[j] = detail::fma(acc_batches[j], pt_batches[j], xsimd::batch<OutputType>(monomials[k]));
          } else {
            const auto coeff = xsimd::batch<OutputType>(monomials[k]);
            acc_batches[j] = {detail::fma(real(acc_batches[j]), pt_batches[j], real(coeff)),
                              detail::fma(imag(acc_batches[j]), pt_batches[j], imag(coeff))};
          }
        });
      }
    }

    // Store results
    detail::unroll_loop<OuterUnrollFactor>(
        [&](const auto j) { acc_batches[j].store(out + i + j * simd_size, out_mode); });
  }

  // Remainder points
  ASSUME((trunc - num_points) < block); // tells the compiler that this loop is at most block
  for (auto i = trunc; i < num_points; ++i) {
    out[i] = horner<N_monomials>(map_func(pts[i]), monomials, monomials_size);
  }
}

} // namespace poly_eval
