#pragma once

#include "macros.h"
#include "utils.h"

#include <cassert>
#include <cstddef>
#include <xsimd/xsimd.hpp>

namespace poly_eval {

//------------------------------------------------------------------------------
// Horner (scalar, one-point)
//------------------------------------------------------------------------------

template <std::size_t N_total = 0, typename OutputType, typename InputType>
ALWAYS_INLINE constexpr OutputType horner(const InputType x, const OutputType *c_ptr,
                                          const std::size_t c_size = 0) noexcept {
    if constexpr (N_total != 0) {
        // Compile-time unrolled Horner on reversed array
        OutputType acc = c_ptr[0];
        detail::unroll_loop<0, N_total>([&]([[maybe_unused]] const auto I) {
            constexpr auto k = decltype(I)::value;
            acc = detail::fma(acc, x, c_ptr[k]);
        });
        return acc;
    } else {
        // Runtime iterative Horner
        OutputType acc = c_ptr[0];
        for (std::size_t k = 1; k < c_size; ++k) {
            acc = detail::fma(acc, x, c_ptr[k]);
        }
        return acc;
    }
}

//------------------------------------------------------------------------------
// SIMD Horner (coeffs reversed)
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
        xsimd::batch<InputType> pt_batches[OuterUnrollFactor];
        xsimd::batch<OutputType> acc_batches[OuterUnrollFactor];

        // Load and init with highest-degree term
        detail::unroll_loop<OuterUnrollFactor>([&]([[maybe_unused]] const auto I) {
            constexpr auto j = decltype(I)::value;
            pt_batches[j] = map_func(xsimd::load(pts + i + j * simd_size, pts_mode));
            acc_batches[j] = xsimd::batch<OutputType>(monomials[0]);
        });

        // Horner steps
        if constexpr (N_monomials != 0) {
            detail::unroll_loop<0, N_monomials>([&]([[maybe_unused]] const auto I) {
                constexpr auto k = decltype(I)::value;
                detail::unroll_loop<OuterUnrollFactor>([&]([[maybe_unused]] const auto J) {
                    constexpr auto j = decltype(J)::value;
                    if constexpr (std::is_floating_point_v<OutputType>) {
                        acc_batches[j] =
                            detail::fma(acc_batches[j], pt_batches[j], xsimd::batch<OutputType>(monomials[k]));
                    } else {
                        auto coeff = xsimd::batch<OutputType>(monomials[k]);
                        acc_batches[j] = {detail::fma(real(acc_batches[j]), pt_batches[j], real(coeff)),
                                          detail::fma(imag(acc_batches[j]), pt_batches[j], imag(coeff))};
                    }
                });
            });
        } else {
            for (std::size_t k = 1; k < monomials_size; ++k) {
                detail::unroll_loop<OuterUnrollFactor>([&]([[maybe_unused]] const auto I) {
                    constexpr auto j = decltype(I)::value;
                    if constexpr (std::is_floating_point_v<OutputType>) {
                        acc_batches[j] =
                            detail::fma(acc_batches[j], pt_batches[j], xsimd::batch<OutputType>(monomials[k]));
                    } else {
                        auto coeff = xsimd::batch<OutputType>(monomials[k]);
                        acc_batches[j] = {detail::fma(real(acc_batches[j]), pt_batches[j], real(coeff)),
                                          detail::fma(imag(acc_batches[j]), pt_batches[j], imag(coeff))};
                    }
                });
            }
        }

        // Store results
        detail::unroll_loop<OuterUnrollFactor>([&]([[maybe_unused]] const auto I) {
            constexpr auto j = decltype(I)::value;
            acc_batches[j].store(out + i + j * simd_size, out_mode);
        });
    }

    // Remainder
    ASSUME((trunc - num_points) < block);
    for (std::size_t idx = trunc; idx < num_points; ++idx) {
        out[idx] = horner<N_monomials>(map_func(pts[idx]), monomials, monomials_size);
    }
}

//------------------------------------------------------------------------------
// horner_many
//------------------------------------------------------------------------------

template <std::size_t M_total = 0, std::size_t N_total = 0, bool scaling = false, typename OutputType,
          typename InputType>
ALWAYS_INLINE constexpr void horner_many(const InputType x, const OutputType *coeffs, OutputType *out,
                                         const std::size_t M = 0, const std::size_t N = 0,
                                         const InputType *low = nullptr, const InputType *hi = nullptr) noexcept {
    const std::size_t m_lim = M_total ? M_total : M;
    const std::size_t n_lim = N_total ? N_total : N;

    if constexpr (M_total != 0) {
        detail::unroll_loop<M_total>([&]([[maybe_unused]] const auto I) {
            constexpr auto m = decltype(I)::value;
            const auto xm = scaling ? (InputType{2} * x - hi[m]) * low[m] : x;
            out[m] = horner<N_total>(xm, coeffs + m * n_lim, n_lim);
        });
    } else {
        for (std::size_t m = 0; m < m_lim; ++m) {
            const auto xm = scaling ? (InputType{2} * x - hi[m]) * low[m] : x;
            out[m] = horner<N_total>(xm, coeffs + m * n_lim, n_lim);
        }
    }
}

//------------------------------------------------------------------------------
// horner_transposed
//------------------------------------------------------------------------------

template <std::size_t M_total = 0, std::size_t N_total = 0, std::size_t simd_width = 0, typename Out, typename In>
ALWAYS_INLINE constexpr void horner_transposed(const In *x, const Out *c, Out *out, const std::size_t M = 0,
                                               const std::size_t N = 0) noexcept {
    constexpr bool has_Mt = (M_total != 0);
    constexpr bool has_Nt = (N_total != 0);
    constexpr bool do_simd = (simd_width > 0);
    const std::size_t m_lim = has_Mt ? M_total : M;
    const std::size_t n_lim = has_Nt ? N_total : N;
    const std::size_t stride = m_lim;

    if constexpr (do_simd) {
        using batch_in = xsimd::make_sized_batch_t<In, simd_width>;
        using batch_out = xsimd::make_sized_batch_t<Out, simd_width>;
        if constexpr (has_Mt) {
            constexpr std::size_t C = M_total / simd_width;
            std::array<batch_out, C> acc;
            std::array<batch_in, C> xvec;

            detail::unroll_loop<C>([&]([[maybe_unused]] const auto I) {
                constexpr auto ci = decltype(I)::value;
                const auto base = ci * simd_width;
                xvec[ci] = batch_in::load_unaligned(x + base);
                acc[ci] = batch_out::load_unaligned(c + base);
            });

            if constexpr (has_Nt) {
                detail::unroll_loop<0, N_total>([&]([[maybe_unused]] const auto I) {
                    constexpr auto k = decltype(I)::value;
                    if constexpr (k > 0) {
                        const auto col = c + k * stride;
                        detail::unroll_loop<C>([&]([[maybe_unused]] const auto I2) {
                            constexpr auto ci = decltype(I2)::value;
                            const auto base = ci * simd_width;
                            batch_out ck = batch_out::load_unaligned(col + base);
                            acc[ci] = detail::fma(acc[ci], xvec[ci], ck);
                        });
                    }
                });
            } else {
                for (std::size_t k = 1; k < n_lim; ++k) {
                    const auto col = c + k * stride;
                    detail::unroll_loop<C>([&]([[maybe_unused]] const auto I) {
                        constexpr auto ci = decltype(I)::value;
                        const auto base = ci * simd_width;
                        batch_out ck = batch_out::load_unaligned(col + base);
                        acc[ci] = detail::fma(acc[ci], xvec[ci], ck);
                    });
                }
            }

            detail::unroll_loop<C>([&]([[maybe_unused]] const auto I) {
                constexpr auto ci = decltype(I)::value;
                const auto base = ci * simd_width;
                acc[ci].store_unaligned(out + base);
            });
        } else {
            const std::size_t chunks = m_lim / simd_width;
            std::vector<batch_out> acc(chunks);
            std::vector<batch_in> xvec(chunks);

            for (std::size_t ci = 0; ci < chunks; ++ci) {
                const auto base = ci * simd_width;
                xvec[ci] = batch_in::load_unaligned(x + base);
                acc[ci] = batch_out::load_unaligned(c + base);
            }
            for (std::size_t k = 1; k < n_lim; ++k) {
                const auto col = c + k * stride;
                for (std::size_t ci = 0; ci < chunks; ++ci) {
                    const auto base = ci * simd_width;
                    batch_out ck = batch_out::load_unaligned(col + base);
                    acc[ci] = detail::fma(acc[ci], xvec[ci], ck);
                }
            }
            for (std::size_t ci = 0; ci < chunks; ++ci) {
                const auto base = ci * simd_width;
                acc[ci].store_unaligned(out + base);
            }
        }
    } else {
        // Scalar path
        if constexpr (has_Mt) {
            detail::unroll_loop<M_total>([&]([[maybe_unused]] const auto I) {
                constexpr auto i = decltype(I)::value;
                out[i] = c[i];
            });
        } else {
            for (std::size_t i = 0; i < m_lim; ++i)
                out[i] = c[i];
        }

        const auto step = [&](const std::size_t k) noexcept {
            const auto col = c + k * stride;
            if constexpr (has_Mt) {
                detail::unroll_loop<M_total>([&]([[maybe_unused]] const auto I) {
                    constexpr auto i = decltype(I)::value;
                    out[i] = detail::fma(out[i], x[i], col[i]);
                });
            } else {
                for (std::size_t i = 0; i < m_lim; ++i)
                    out[i] = detail::fma(out[i], x[i], col[i]);
            }
        };

        if constexpr (has_Nt) {
            detail::unroll_loop<0, N_total>([&]([[maybe_unused]] const auto I) {
                constexpr auto k = decltype(I)::value;
                if constexpr (k > 0)
                    step(k);
            });
        } else {
            for (std::size_t k = 1; k < n_lim; ++k)
                step(k);
        }
    }
}

//------------------------------------------------------------------------------
// detail::horner_1d
//------------------------------------------------------------------------------

namespace detail {

template <std::size_t DegCT = 0, typename OutT, typename InScalar>
ALWAYS_INLINE constexpr OutT horner_1d(const InScalar x, const OutT *c_ptr, const std::size_t deg_rt = 0) noexcept {
    const std::size_t deg = DegCT ? DegCT : deg_rt;
    OutT acc = c_ptr[0];
    if constexpr (DegCT != 0) {
        detail::unroll_loop<0, DegCT - 1>([&]([[maybe_unused]] const auto I) {
            constexpr auto k = decltype(I)::value;
            acc = detail::fma(acc, x, c_ptr[k]);
        });
    } else {
        for (std::size_t k = 1; k < deg; ++k)
            acc = detail::fma(acc, x, c_ptr[k]);
    }
    return acc;
}

// helper to invoke coeffs(idx[0],…,idx[D‑1], d) in C++17
template <std::size_t D, typename MdspanType, std::size_t... Is>
ALWAYS_INLINE constexpr auto call_coeffs_impl(const MdspanType &coeffs, const std::array<std::size_t, D> &idx,
                                              std::index_sequence<Is...>, std::size_t d) noexcept {
    return coeffs(idx[Is]..., d);
}

template <std::size_t D, typename MdspanType>
ALWAYS_INLINE constexpr auto call_coeffs(const MdspanType &coeffs, const std::array<std::size_t, D> &idx,
                                         std::size_t d) noexcept {
    return call_coeffs_impl<D, MdspanType>(coeffs, idx, std::make_index_sequence<D>{}, d);
}

//------------------------------------------------------------------------------
// detail::horner_nd_impl
//------------------------------------------------------------------------------

template <std::size_t Level, std::size_t Dim, std::size_t DegCT, typename OutT, typename InVec, typename Mdspan>
ALWAYS_INLINE constexpr OutT horner_nd_impl(const InVec &x, const Mdspan &coeffs, std::array<std::size_t, Dim> &idx,
                                            const int deg_rt) {
    constexpr std::size_t axis = Dim - Level;
    constexpr std::size_t OUT = std::tuple_size_v<OutT>;
    const int deg = DegCT ? static_cast<int>(DegCT) : deg_rt;

    using batch =
        xsimd::make_sized_batch_t<typename OutT::value_type, optimal_simd_width<typename OutT::value_type, OUT>()>;
    alignas(batch::arch_type::alignment()) OutT res{0};
    const batch x_vec(x[axis]);

    for (int k = 0; k < deg; ++k) {
        idx[axis] = static_cast<std::size_t>(k);
        alignas(batch::arch_type::alignment()) OutT inner{};
        if constexpr (Level > 1) {
            inner = horner_nd_impl<Level - 1, Dim, DegCT, OutT>(x, coeffs, idx, deg_rt);
        } else {
            detail::unroll_loop<OUT>([&]([[maybe_unused]] const auto I) {
                constexpr auto d = decltype(I)::value;
                inner[d] = call_coeffs<Dim>(coeffs, idx, d);
            });
        }

        detail::unroll_loop<0, OUT, /*Inc=*/batch::size>([&]([[maybe_unused]] const auto I) {
            constexpr auto d = decltype(I)::value;
            if constexpr (d + batch::size <= OUT) {
                auto in = batch::load_aligned(inner.data() + d);
                auto r = batch::load_aligned(res.data() + d);
                detail::fma(r, x_vec, in).store_aligned(res.data() + d);
            } else {
                detail::unroll_loop<d, OUT>([&]([[maybe_unused]] const auto J) {
                    constexpr auto last = decltype(J)::value;
                    res[last] = detail::fma(res[last], typename OutT::value_type(x[axis]), inner[last]);
                });
            }
        });
    }
    return res;
}

} // namespace detail

//------------------------------------------------------------------------------
// Front-ends
//------------------------------------------------------------------------------

template <std::size_t DegCT = 0, typename OutT, typename InVec, typename Mdspan>
ALWAYS_INLINE constexpr OutT horner(const InVec &x, const Mdspan &coeffs, int deg_rt) {
    constexpr std::size_t Dim = Mdspan::rank() - 1;
    std::array<std::size_t, Dim> idx{};
    return detail::horner_nd_impl<Dim, Dim, DegCT, OutT>(x, coeffs, idx, deg_rt);
}

} // namespace poly_eval