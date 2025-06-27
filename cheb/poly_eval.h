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
    // Compile-time unrolled Horner on reversed array
    // Start with highest-degree term at c_ptr[0]
    OutputType acc = c_ptr[0];
    detail::unroll_loop<N_total, 1>([&]<std::size_t k>() {
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
//  horner_transposed – column-major, reversed Horner
//==============================================================================
//
//  •  M_total (#polynomials)    0 → run-time M
//  •  N_total (#coefficients)   0 → run-time N
//
//  Both loops are fully un-rolled when the corresponding template
//  parameter is non-zero.
//------------------------------------------------------------------------------
template <std::size_t M_total = 0,                                // #polynomials (0 → run-time)
          std::size_t N_total = 0,                                // degree        (0 → run-time)
          typename Out,                                           // OutputType
          typename In>                                            // InputType
ALWAYS_INLINE constexpr void horner_transposed(const In *x,       // [M] scaled inputs
                                               const Out *c,      // [M*N] coeffs (col-major, reversed)
                                               Out *out,          // [M] results
                                               std::size_t M = 0, // run-time M
                                               std::size_t N = 0) // run-time N
    noexcept {
  const std::size_t m_lim = M_total ? M_total : M;
  const std::size_t n_lim = N_total ? N_total : N;
  const std::size_t stride = m_lim; // elems per column

  // –––––––––––– initialise accumulators with degree-0 column ––––––––––––
  if constexpr (M_total != 0) {
    detail::unroll_loop<M_total>([&]<std::size_t i>() constexpr { out[i] = c[i]; });
  } else {
    for (std::size_t i = 0; i < m_lim; ++i)
      out[i] = c[i];
  }

  // helper for one degree step (used in both branches below)
  const auto step_degree = [&](const std::size_t k) {
    const Out *col = c + k * stride;
    if constexpr (M_total != 0) {
      detail::unroll_loop<M_total>([&]<std::size_t i>() constexpr { out[i] = detail::fma(out[i], x[i], col[i]); });
    } else {
      for (std::size_t i = 0; i < m_lim; ++i)
        out[i] = detail::fma(out[i], x[i], col[i]);
    }
  };

  // –––––––––––– Horner recursion over remaining degrees ––––––––––––––––
  if constexpr (N_total != 0) {
    detail::unroll_loop<N_total, 1>([&]<std::size_t k>() constexpr { step_degree(k); });
  } else {
    for (std::size_t k = 1; k < n_lim; ++k) {
      step_degree(k);
    }
  }
}

/*
   // --------------------------- size bookkeeping ------------------------------
const std::size_t m_lim = M_total ? M_total : M;
const std::size_t n_lim = N_total ? N_total : N;

// Runtime UNROLL (ignored if template UNROLL > 0)
const std::size_t UN = (UNROLL > 0) ? UNROLL
                                    : ((unroll_rt > 0) ? unroll_rt : 1);

// --------------------------- helpers ---------------------------------------
auto step_fma = [&](auto& acc, const OutputType& c) {
  if constexpr (std::is_floating_point_v<OutputType>) {
    acc = detail::fma(acc, x, c);
  } else {          // complex / struct with .real/.imag
    acc = OutputType{detail::fma(real(acc), x, real(c)),
                     detail::fma(imag(acc), x, imag(c))};
  }
};

// ----------------------- main evaluation loop -----------------------------
const std::size_t m_trunc = m_lim / UN * UN;   // largest multiple of UN

// ---- batched section ------------------------------------------------------
for (std::size_t base = 0; base < m_trunc; base += UN) {

  // ---- initialise accumulators with highest-degree term -------------------
  OutputType acc[UN];
  if constexpr (UNROLL > 0) {
    detail::unroll_loop<UNROLL>([&]<std::size_t j>() {
      acc[j] = coeffs[(base + j) * n_lim];
    });
  } else {
    for (std::size_t j = 0; j < UN; ++j)
      acc[j] = coeffs[(base + j) * n_lim];
  }

  // ---- Horner iterations --------------------------------------------------
  if constexpr (N_total > 0) {
    detail::unroll_loop<N_total, 1>([&]<std::size_t k>() {
      if constexpr (UNROLL > 0) {
        detail::unroll_loop<UNROLL>([&]<std::size_t j>() {
          step_fma(acc[j], coeffs[(base + j) * n_lim + k]);
        });
      } else {
        for (std::size_t j = 0; j < UN; ++j)
          step_fma(acc[j], coeffs[(base + j) * n_lim + k]);
      }
    });
  } else {
    for (std::size_t k = 1; k < n_lim; ++k) {
      if constexpr (UNROLL > 0) {
        detail::unroll_loop<UNROLL>([&]<std::size_t j>() {
          step_fma(acc[j], coeffs[(base + j) * n_lim + k]);
        });
      } else {
        for (std::size_t j = 0; j < UN; ++j)
          step_fma(acc[j], coeffs[(base + j) * n_lim + k]);
      }
    }
  }

  // ---- write results ------------------------------------------------------
  if constexpr (UNROLL > 0) {
    detail::unroll_loop<UNROLL>([&]<std::size_t j>() {
      out[base + j] = acc[j];
    });
  } else {
    for (std::size_t j = 0; j < UN; ++j)
      out[base + j] = acc[j];
  }
}

// ---- tail ( < UN polynomials ) -------------------------------------------
for (std::size_t m = m_trunc; m < m_lim; ++m)
  out[m] = horner<N_total>(x, coeffs + m * n_lim, n_lim);
  */
// }
} // namespace poly_eval
