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
} // namespace std
#endif

namespace poly_eval {

// -----------------------------------------------------------------------------
// Buffer: Conditional type alias for std::vector or std::array
// -----------------------------------------------------------------------------
template <typename T, std::size_t N_compile_time_val>
using Buffer = std::conditional_t<N_compile_time_val == 0, std::vector<T>, std::array<T, N_compile_time_val>>;

template <typename T, std::size_t N_compile_time_val, std::size_t alignment>
using AlignedBuffer = std::conditional_t < N_compile_time_val == 0 &&
                      N_compile_time_val<16384, std::vector<T, xsimd::aligned_allocator<T, alignment>>,
                                         std::array<T, N_compile_time_val>>;

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
    return xsimd::fma(a, b, c);
}

template <typename T, typename U> constexpr ALWAYS_INLINE T fma(const T &a, const U &b, const T &c) noexcept {
    // Fused multiply-add: a * b + c
    return T{fma(real(a), b, real(c)), fma(imag(a), b, imag(c))};
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

// – your range‑count helper (unchanged)
template <std::size_t Start, std::size_t Stop, std::size_t Inc>
inline constexpr std::size_t compute_range_count = (Start < Stop ? ((Stop - Start + Inc - 1) / Inc) : 0);

// – the unroll implementation that feeds you integral_constant<I>
template <std::size_t Start, std::size_t Inc, typename F, std::size_t... Is>
constexpr void unroll_loop_impl(F &&f, std::index_sequence<Is...>) {
    // Calls f(std::integral_constant<std::size_t, Start + Is*Inc>{}) for each Is
    (f(std::integral_constant<std::size_t, Start + Is * Inc>{}), ...);
}

// – your user‑facing function, same name as before
template <std::size_t Start, std::size_t Stop, std::size_t Inc = 1, typename F> constexpr void unroll_loop(F &&f) {
    constexpr std::size_t Count = compute_range_count<Start, Stop, Inc>;
    unroll_loop_impl<Start, Inc>(std::forward<F>(f), std::make_index_sequence<Count>{});
}

template <std::size_t Stop,typename F> constexpr void unroll_loop(F &&f) {
    return unroll_loop<0, Stop, 1>(std::forward<F>(f));
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

// stand-alone Bjorck–Pereyra divided‐difference
template <std::size_t N, class X, class Y>
C20CONSTEXPR Buffer<Y, N> bjorck_pereyra(const Buffer<X, N> &x, const Buffer<Y, N> &y) {
    const std::size_t n = (N == 0 ? x.size() : N);
    Buffer<Y, N> a = y;
    for (std::size_t k = 0; k + 1 < n; ++k) {
        for (std::size_t i = n - 1; i >= k + 1; --i) {
            a[i] = (a[i] - a[i - 1]) / static_cast<Y>(x[i] - x[i - k - 1]);
        }
    }
    return a;
}

// stand-alone Newton→monomial conversion
template <std::size_t N, class X, class Y>
C20CONSTEXPR Buffer<Y, N> newton_to_monomial(const Buffer<Y, N> &alpha, const Buffer<X, N> &nodes) {
    int n = static_cast<int>(alpha.size());
    Buffer<Y, N * 2> c{0};
    if constexpr (N == 0) {
        c.reserve(n);
        c.push_back(Y(0));
    }
    std::size_t deg = 0;
    for (int i = n - 1; i >= 0; --i) {
        ++deg;
        if constexpr (N == 0)
            c.push_back(Y(0));
        for (int j = static_cast<int>(deg); j >= 1; --j) {
            c[j] = c[j - 1] - nodes[i] * c[j];
        }
        c[0] = -nodes[i] * c[0] + alpha[i];
    }
    if constexpr (N == 0) {
        if (static_cast<int>(c.size()) > n)
            c.resize(n);
        return c;
    }
    Buffer<Y, N> result{};
    std::copy_n(c.begin(), N, result.begin());
    return result;
}

template <class T, std::size_t N = 1> constexpr uint8_t min_simd_width() {
    if constexpr (std::is_void_v<xsimd::make_sized_batch_t<T, N>>) {
        return min_simd_width<T, N * 2>();
    } else {
        return N;
    }
}

template <class T, std::size_t Upper, std::size_t Width> constexpr std::size_t optimal_impl() {
    if constexpr (Width * 2 <= Upper && !std::is_void_v<xsimd::make_sized_batch_t<T, Width * 2>>) {
        return optimal_impl<T, Upper, Width * 2>();
    } else {
        return Width;
    }
}

template <class T, std::size_t N> constexpr std::size_t optimal_simd_width() {
    constexpr uint8_t arch_max = xsimd::batch<T>::size;
    constexpr uint8_t upper = (N < arch_max) ? N : arch_max;
    constexpr uint8_t start = min_simd_width<T>();
    if constexpr (start > upper) {
        // N is smaller than the smallest SIMD batch → scalar
        return start;
    } else {
        return optimal_impl<T, upper, start>();
    }
}

// --- Helper to create static extents when N_compile > 0 --------------------
// 1) free helper: build a static extents type for N_compile > 0
template <std::size_t N_compile, std::size_t Dim, std::size_t OutDim, std::size_t... Is>
constexpr auto make_static_extents_impl(std::index_sequence<Is...>) {
    // expands to extents<N_compile, N_compile, …, OutDim>
    return stdex::extents<std::size_t, ((void)Is, N_compile)..., OutDim>{};
}

template <std::size_t N_compile, std::size_t Dim, std::size_t OutDim>
using static_extents_t = decltype(make_static_extents_impl<N_compile, Dim, OutDim>(std::make_index_sequence<Dim>{}));

// 2) free helper: compute how many entries that layout_right mdspan needs
template <class Scalar, std::size_t N_compile, std::size_t Dim, std::size_t OutDim>
constexpr std::size_t storage_required() {
    // pick the extents type directly (no const!)
    using extents_t = static_extents_t<N_compile, Dim, OutDim>;
    using mdspan_t = stdex::mdspan<Scalar, extents_t, stdex::layout_right>;
    // default‑construct an extents_t and query its mapping
    return typename mdspan_t::mapping_type{extents_t{}}.required_span_size();
}

template <std::size_t N_compile, std::size_t DimIn, std::size_t DimOut, std::size_t... Is>
constexpr auto make_static_extents(std::index_sequence<Is...>) {
    return stdex::extents<std::size_t, ((void)Is, N_compile)..., DimOut>{};
}

} // namespace poly_eval::detail