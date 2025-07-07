#pragma once

#include "macros.h"
#include "utils.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <xsimd/xsimd.hpp>

namespace poly_eval {

template <std::size_t N_total = 0, typename OutputType, typename InputType>
ALWAYS_INLINE constexpr OutputType horner(const InputType x, const OutputType *c_ptr, std::size_t c_size = 0) noexcept {
  if constexpr (N_total != 0) {
    // Compile-time unrolled Horner on reversed array
    // Start with highest-degree term at c_ptr[0]
    OutputType acc = c_ptr[0];
    detail::unroll_loop<N_total, 1>([&]<std::size_t k>() { acc = detail::fma(acc, x, c_ptr[k]); });
    return acc;
  } else {
    // Runtime iterative Horner on reversed array
    OutputType acc = c_ptr[0];
    for (std::size_t k = 1; k < c_size; ++k) {
      acc = detail::fma(acc, x, c_ptr[k]);
    }
    return acc;
  }
}

//------------------------------------------------------------------------------
// SIMD Horner (coefficients reversed)
//------------------------------------------------------------------------------

template <std::size_t N_monomials = 0, bool pts_aligned = false, bool out_aligned = false, int UNROLL = 0,
          typename InputType, typename OutputType, typename MapFunc>
ALWAYS_INLINE constexpr void horner(
    const InputType *pts, OutputType *out, std::size_t num_points, const OutputType *monomials,
    std::size_t monomials_size, const MapFunc map_func = [](auto v) { return v; }) noexcept {

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
    detail::unroll_loop<OuterUnrollFactor>([&]<std::size_t j>() {
      pt_batches[j] = map_func(xsimd::load(pts + i + j * simd_size, pts_mode));
      acc_batches[j] = xsimd::batch<OutputType>(monomials[0]);
    });

    // Horner steps: ascending coefficients
    if constexpr (N_monomials != 0) {
      // Compile-time unroll over coefficients
      detail::unroll_loop<N_monomials, 1>([&]<std::size_t k>() {
        detail::unroll_loop<OuterUnrollFactor>([&]<std::size_t j>() {
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
      // Runtime fallback
      for (std::size_t k = 1; k < monomials_size; ++k) {
        detail::unroll_loop<OuterUnrollFactor>([&]<std::size_t j>() {
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
        [&]<std::size_t j>() { acc_batches[j].store(out + i + j * simd_size, out_mode); });
  }

  // Remainder points
  ASSUME((trunc - num_points) < block); // tells the compiler that this loop is at most block
  for (auto i = trunc; i < num_points; ++i) {
    out[i] = horner<N_monomials>(map_func(pts[i]), monomials, monomials_size);
  }
}

//------------------------------------------------------------------------------
//  horner_many  — scalar, one-point / many-polynomials
//------------------------------------------------------------------------------
// • Coeff layout (reversed):  c00 … c0(N-1),  c10 … c1(N-1), … , c(M-1)0 …
// • M_total / N_total = 0 ⇒ size comes from run-time args M, N
//------------------------------------------------------------------------------
template <std::size_t M_total = 0, // #polynomials (0 → run-time)
          std::size_t N_total = 0, // #coeffs each   (0 → run-time)
          bool scaling = false,    // if true, apply per-poly scaling
          typename OutputType,
          typename InputType>
ALWAYS_INLINE constexpr void horner_many(const InputType x,              // input array (size = 1)
                                         const OutputType *coeffs,       // coeff array (size ≥ M*N)
                                         OutputType *out,                // output array (size ≥ M)
                                         std::size_t M = 0,              // run-time #polynomials
                                         std::size_t N = 0,              // run-time #coeffs
                                         const InputType *low = nullptr, // optional scaling params
                                         const InputType *hi = nullptr)  // optional scaling params
    noexcept {
  // how many polys / coeffs?
  const std::size_t m_lim = M_total ? M_total : M;
  const std::size_t n_lim = N_total ? N_total : N;

  if constexpr (M_total != 0) {
    // compile-time unroll over [0..M_total)
    detail::unroll_loop<M_total>([&]<std::size_t m>() {
      const auto xm = scaling ? (InputType{2} * x - hi[m]) * low[m] : x;
      out[m] = horner<N_total>(xm, coeffs + m * n_lim, n_lim);
    });
  } else {
    // run-time loop
    for (std::size_t m = 0; m < m_lim; ++m) {
      const auto xm = scaling ? (InputType{2} * x - hi[m]) * low[m] : x;
      out[m] = horner<N_total>(xm, coeffs + m * n_lim, n_lim);
    }
  }
}

//==============================================================================
//  horner_transposed – column-major, reversed Horner, optional SIMD width
//==============================================================================
//
//  •  M_total (#polynomials)    0 → run-time M
//  •  N_total (#coefficients)   0 → run-time N
//  •  simd_width                0 → no vectorization, otherwise must divide M
//
//  Both loops are fully un-rolled when the corresponding template
//  parameter is non-zero. If simd_width>0 we process M in blocks of simd_width
//  using xsimd::make_sized_batch_t<In,simd_width>, and when M_total>0 we also
//  unroll those chunk loops via detail::unroll_loop.
//
template <std::size_t M_total = 0,    // #polynomials (0 → run‑time M)
          std::size_t N_total = 0,    // degree        (0 → run‑time N)
          std::size_t simd_width = 0, // 0 → scalar, >0 → use xsimd batches
          typename Out,
          typename In>
ALWAYS_INLINE constexpr void horner_transposed(const In *x,       // [M] scaled inputs (one per polynomial)
                                               const Out *c,      // [M*N] coeffs (col‑major, reversed)
                                               Out *out,          // [M]   results
                                               std::size_t M = 0, // run‑time M
                                               std::size_t N = 0  // run‑time N
                                               ) noexcept {
  constexpr bool has_Mt = (M_total != 0);
  constexpr bool has_Nt = (N_total != 0);
  constexpr bool do_simd = (simd_width > 0);

  const std::size_t m_lim = has_Mt ? M_total : M;
  const std::size_t n_lim = has_Nt ? N_total : N;
  const std::size_t stride = m_lim; // elems per column

  if constexpr (do_simd) {
    //------------------------------------------------------------------------
    // SIMD path – keep everything in registers for all degrees
    //------------------------------------------------------------------------
    using batch_in = xsimd::make_sized_batch_t<In, simd_width>;
    using batch_out = xsimd::make_sized_batch_t<Out, simd_width>;

    // compile‑time guard
    static_assert(!has_Mt || (M_total % simd_width == 0), "M_total must be a multiple of simd_width when simd_width>0");

    //------------------------------------------------------------------
    // 1. Allocate per‑chunk accumulators and (optionally) cached x
    //------------------------------------------------------------------
    if constexpr (has_Mt) {
      // Compile‑time number of chunks → use std::array and unroll
      constexpr std::size_t C = M_total / simd_width;
      std::array<batch_out, C> acc{};
      std::array<batch_in, C> xvec{};

      // Load x once and initialise acc with degree‑0 column
      detail::unroll_loop<C>([&]<std::size_t ci>() constexpr {
        constexpr std::size_t base = ci * simd_width;
        xvec[ci] = batch_in ::load_unaligned(x + base);
        acc[ci] = batch_out::load_unaligned(c + base); // k = 0
      });

      //------------------------------------------------------------------
      // 2. Horner recursion over remaining degrees (k = 1 … n_lim‑1)
      //------------------------------------------------------------------
      if constexpr (has_Nt) {
        detail::unroll_loop<N_total, 1>([&]<std::size_t k>() constexpr {
          if constexpr (k > 0) {
            constexpr std::size_t col_offset = k * stride;
            detail::unroll_loop<C>([&]<std::size_t ci>() constexpr {
              constexpr std::size_t base = ci * simd_width;
              batch_out ck = batch_out::load_unaligned(c + col_offset + base);
              acc[ci] = detail::fma(acc[ci], xvec[ci], ck);
            });
          }
        });
      } else {
        for (std::size_t k = 1; k < n_lim; ++k) {
          const Out *col = c + k * stride;
          detail::unroll_loop<C>([&]<std::size_t ci>() constexpr {
            constexpr std::size_t base = ci * simd_width;
            batch_out ck = batch_out::load_unaligned(col + base);
            acc[ci] = detail::fma(acc[ci], xvec[ci], ck);
          });
        }
      }

      //------------------------------------------------------------------
      // 3. Final store of the accumulators into `out`
      //------------------------------------------------------------------
      detail::unroll_loop<C>([&]<std::size_t ci>() constexpr {
        constexpr std::size_t base = ci * simd_width;
        acc[ci].store_unaligned(out + base);
      });
    } else {

      const std::size_t chunks = m_lim / simd_width;

      // Runtime number of chunks → use std::vector + normal loops
      std::vector<batch_out> acc(chunks);
      std::vector<batch_in> xvec(chunks);

      // Load x once and initialise acc with degree‑0 column
      for (std::size_t ci = 0; ci < chunks; ++ci) {
        std::size_t base = ci * simd_width;
        xvec[ci] = batch_in ::load_unaligned(x + base);
        acc[ci] = batch_out::load_unaligned(c + base);
      }

      // Horner recursion – keep everything in registers
      for (std::size_t k = 1; k < n_lim; ++k) {
        const Out *col = c + k * stride;
        for (std::size_t ci = 0; ci < chunks; ++ci) {
          std::size_t base = ci * simd_width;
          batch_out ck = batch_out::load_unaligned(col + base);
          acc[ci] = detail::fma(acc[ci], xvec[ci], ck);
        }
      }

      // Single final store
      for (std::size_t ci = 0; ci < chunks; ++ci) {
        std::size_t base = ci * simd_width;
        acc[ci].store_unaligned(out + base);
      }
    }
  } else {
    //------------------------------------------------------------------------
    // Scalar path (original implementation)
    //------------------------------------------------------------------------
    // 1. Initialise accumulators with degree‑0 column
    if constexpr (has_Mt) {
      detail::unroll_loop<M_total>([&]<std::size_t i>() constexpr { out[i] = c[i]; });
    } else {
      for (std::size_t i = 0; i < m_lim; ++i)
        out[i] = c[i];
    }

    // 2. Horner recursion – scalar
    const auto step = [&](std::size_t k) noexcept {
      const Out *col = c + k * stride;
      if constexpr (has_Mt) {
        detail::unroll_loop<M_total>([&]<std::size_t i>() constexpr { out[i] = detail::fma(out[i], x[i], col[i]); });
      } else {
        for (std::size_t i = 0; i < m_lim; ++i)
          out[i] = detail::fma(out[i], x[i], col[i]);
      }
    };

    if constexpr (has_Nt) {
      detail::unroll_loop<N_total, 1>([&]<std::size_t k>() constexpr {
        if constexpr (k > 0)
          step(k);
      });
    } else {
      for (std::size_t k = 1; k < n_lim; ++k)
        step(k);
    }
  }
}

} // namespace poly_eval
