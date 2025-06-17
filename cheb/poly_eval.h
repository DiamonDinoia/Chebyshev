#pragma once

#include "macros.h"
#include "utils.h"

#include <cassert>
#include <xsimd/xsimd.hpp>
#include <cstddef>


namespace poly_eval {

// -----------------------------------------------------------------------------
// PolyEval: standalone Horner polynomial evaluator
// -----------------------------------------------------------------------------

template <typename T,
          std::size_t N_compile_time = 0>
class PolyEval {
public:
  // Runtime-sized constructor (only when N_compile_time == 0)
  template <std::size_t N = N_compile_time,
            typename = std::enable_if_t<N == 0>>
  C20CONSTEXPR PolyEval(const T *coeffs, int n)
    : _coeffs(coeffs), _n_terms(n) {
    assert(_n_terms > 0 && "Polynomial degree must be positive");
  }

  // Compile-time-sized constructor (only when N_compile_time > 0)
  template <std::size_t N = N_compile_time,
            typename = std::enable_if_t<(N > 0)>>
  C20CONSTEXPR PolyEval(const T *coeffs)
    : _coeffs(coeffs), _n_terms(static_cast<int>(N_compile_time)) {
    static_assert(N_compile_time > 0, "Need compile-time size > 0");
  }

  // Single-point Horner evaluation (expects x already in Chebyshev domain)
  C20CONSTEXPR T eval(T x) const noexcept {
    if constexpr (N_compile_time > 0) {
      return horner<N_compile_time, 0>(_coeffs, x);
    } else {
      T acc = _coeffs[_n_terms - 1];
      for (int k = _n_terms - 2; k >= 0; --k) {
        acc = std::fma(x, acc, _coeffs[k]);
      }
      return acc;
    }
  }

  // Scalar call operator delegates to eval
  C20CONSTEXPR T operator()(T x) const noexcept { return eval(x); }

  // Batch evaluation (SIMD + unrolling) - inputs must be pre-mapped
  template <int OuterUnrollFactor, bool pts_aligned, bool out_aligned>
  ALWAYS_INLINE
  void eval_batch(const T * RESTRICT pts,
                  T * RESTRICT out,
                  std::size_t num_points) const noexcept {
    static_assert(OuterUnrollFactor > 0 && (OuterUnrollFactor & (OuterUnrollFactor - 1)) == 0,
                  "OuterUnrollFactor must be a power of two greater than zero.");

    constexpr auto simd_size = xsimd::batch<T>::size;
    const auto trunc_size = num_points & (-(int)(simd_size * OuterUnrollFactor));

    // Determine aligned modes
    static constexpr auto pts_mode = [] {
      if constexpr (pts_aligned)
        return xsimd::aligned_mode{};
      else
        return xsimd::unaligned_mode{};
    }();
    static constexpr auto out_mode = [] {
      if constexpr (out_aligned)
        return xsimd::aligned_mode{};
      else
        return xsimd::unaligned_mode{};
    }();

    // Process full SIMD+unroll blocks
    for (std::size_t i = 0; i < trunc_size; i += simd_size * OuterUnrollFactor) {
      xsimd::batch<T> pt_batches[OuterUnrollFactor];
      xsimd::batch<T> acc_batches[OuterUnrollFactor];

      // Load & initialize
      detail::unroll_loop<OuterUnrollFactor>([&](auto j) {
        pt_batches[j] = xsimd::load(pts + i + j * simd_size, pts_mode);
        acc_batches[j] = xsimd::batch<T>(_coeffs[size() - 1]);
      });

      // Horner inner (compile-time or runtime)
      if constexpr (N_compile_time > 0) {
        horner<N_compile_time - 1, 0, OuterUnrollFactor>(pt_batches, acc_batches);
      } else {
        for (int k = _n_terms - 2; k >= 0; --k) {
          detail::unroll_loop<OuterUnrollFactor>([&](auto j) {
            acc_batches[j] = xsimd::fma(pt_batches[j], acc_batches[j], xsimd::batch<T>(_coeffs[k]));
          });
        }
      }

      // Store
      detail::unroll_loop<OuterUnrollFactor>([&](auto j) {
        xsimd::store(out + i + j * simd_size, acc_batches[j], out_mode);
      });
    }

    // Remainder
    for (std::size_t i = trunc_size; i < num_points; ++i) {
      out[i] = eval(pts[i]);
    }
  }


  // Default batch call operator (uses unaligned, factor=4)

  void operator()(const T *in, T *out, std::size_t num_points) const noexcept {
    eval_batch<4, false, false>(in, out, num_points);
  }


  // Total coefficient count
  constexpr int size() const noexcept {
    return N_compile_time > 0 ? static_cast<int>(N_compile_time) : _n_terms;
  }

private:
  const T *_coeffs;
  int _n_terms = N_compile_time;

  // Recursive Horner for compile-time N
  template <int K_Current, int K_Target, int OuterUnrollFactor>
  ALWAYS_INLINE
  void horner(xsimd::batch<T> * RESTRICT pt_batches,
              xsimd::batch<T> * RESTRICT acc_batches) const noexcept {
    if constexpr (K_Current >= K_Target) {
      [&]<std::size_t... J>(std::integer_sequence<std::size_t, J...>) {
        ((acc_batches[J] = xsimd::fma(
              pt_batches[J], acc_batches[J],
              xsimd::batch<T>(_coeffs[K_Current]))), ...);
      }(std::make_integer_sequence<std::size_t, OuterUnrollFactor>{});
      horner<K_Current - 1, K_Target, OuterUnrollFactor>(pt_batches, acc_batches);
    }
  }


  // Compile-time scalar Horner
  template <std::size_t N_total, std::size_t idx>
  constexpr T horner(const T * RESTRICT c_ptr, T x) const noexcept {
    if constexpr (idx == N_total - 1) {
      return c_ptr[idx];
    } else {
      return std::fma(horner<N_total, idx + 1>(c_ptr, x), x, c_ptr[idx]);
    }
  }

};

} // namespace poly_eval