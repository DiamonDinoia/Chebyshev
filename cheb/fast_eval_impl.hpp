#pragma once

#include "macros.h"
#include "poly_eval.h"
#include "utils.h"

#include <cassert>
#include <iomanip>
#include <iostream>
#include <xsimd/xsimd.hpp>

namespace poly_eval {

namespace detail {
template <typename T> std::vector<T> linspace(const T start, const T end, const int num_points) {
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
    : n_terms(n), low(InputType(1) / (b - a)), hi(b + a) {
    assert(n_terms > 0 && "Polynomial degree must be positive");
    monomials.resize(n_terms);
    initialize_monomials(F, pts);
}

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
template <std::size_t CurrentN, typename>
C20CONSTEXPR FuncEval<Func, N_compile_time, Iters_compile_time>::FuncEval(Func F, const InputType a, const InputType b,
                                                                          const InputType *pts)
    : n_terms(static_cast<int>(CurrentN)), low(InputType(1) / (b - a)), hi(b + a) {
    assert(n_terms > 0 && "Polynomial degree must be positive (template N > 0)");
    initialize_monomials(F, pts);
}

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
typename FuncEval<Func, N_compile_time, Iters_compile_time>::OutputType constexpr ALWAYS_INLINE
FuncEval<Func, N_compile_time, Iters_compile_time>::operator()(const InputType pt) const noexcept {
    const auto xi = map_from_domain(pt);
    return horner<N_compile_time>(xi, monomials.data(), monomials.size()); // Pass data pointer and size
}

// Batch evaluation implementation using SIMD and unrolling
template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
template <int OuterUnrollFactor, bool pts_aligned, bool out_aligned>
ALWAYS_INLINE constexpr void FuncEval<Func, N_compile_time, Iters_compile_time>::horner_polyeval(
    const InputType *RESTRICT pts, OutputType *RESTRICT out, std::size_t num_points) const noexcept {
    static_assert(OuterUnrollFactor >= 0 && (OuterUnrollFactor & (OuterUnrollFactor - 1)) == 0,
                  "OuterUnrollFactor must be a power of two greater than zero.");
    return horner<N_compile_time, pts_aligned, out_aligned, OuterUnrollFactor>(
        pts, out, num_points, monomials.data(), monomials.size(), [this](const auto v) { return map_from_domain(v); });
}

// MUST be defined in a c++ source file
// This is a workaround for the compiler to not the inline the function passed to it.
template <typename F, typename... Args> NO_INLINE static auto noinline(F &&f, Args &&...args) {
    return std::forward<F>(f)(std::forward<Args>(args)...);
}

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
template <bool pts_aligned, bool out_aligned>
ALWAYS_INLINE void constexpr FuncEval<Func, N_compile_time, Iters_compile_time>::operator()(
    const InputType * RESTRICT pts, OutputType * RESTRICT out, std::size_t num_points) const noexcept {
    // find out the alignment of pts and out
    constexpr auto simd_size = xsimd::batch<InputType>::size;
    constexpr auto alignment = xsimd::batch<InputType>::arch_type::alignment();
    constexpr auto unroll_factor = 0;

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
        if (pts_alignment != out_alignment) [[unlikely]] {
            if (pts_alignment >= alignment) [[unlikely]] {
                return noinline([this, pts, out, num_points] {
                    return horner_polyeval<unroll_factor, true, false>(pts, out, num_points);
                });
            }
            if (out_alignment >= alignment) [[unlikely]] {
                return noinline([this, pts, out, num_points] {
                    return horner_polyeval<unroll_factor, false, true>(pts, out, num_points);
                });
            }
            return noinline([this, pts, out, num_points] {
                return horner_polyeval<unroll_factor, false, false>(pts, out, num_points);
            });
        }

        const auto unaligned_points = std::min(
            ((alignment - pts_alignment) & (alignment - 1)) >> detail::countr_zero(sizeof(InputType)), num_points);

        constexpr std::size_t min_align = alignof(std::max_align_t); // in bytes, typically 16
        constexpr std::size_t scalar_unroll = (alignment - min_align) / sizeof(InputType);

        // print alignment;
        ASSUME(unaligned_points < scalar_unroll); // tells the compiler that this loop is at most alignment
        // process scalar until we reach the first aligned point
        detail::unroll_loop<scalar_unroll>([&]<const auto i>() {
            if (i < unaligned_points) {
                out[i] = operator()(pts[i]);
            }
        });
        return horner_polyeval<unroll_factor, true, true>(pts + unaligned_points, out + unaligned_points,
                                                          num_points - unaligned_points);
    }
}

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
C20CONSTEXPR const Buffer<typename FuncEval<Func, N_compile_time, Iters_compile_time>::OutputType, N_compile_time> &
FuncEval<Func, N_compile_time, Iters_compile_time>::coeffs() const noexcept {
    return monomials;
}

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
template <class T>
ALWAYS_INLINE constexpr T
FuncEval<Func, N_compile_time, Iters_compile_time>::map_to_domain(const T T_arg) const noexcept {
    return static_cast<T>(0.5 * (T_arg / low + hi));
}

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
template <class T>
ALWAYS_INLINE constexpr T
FuncEval<Func, N_compile_time, Iters_compile_time>::map_from_domain(const T T_arg) const noexcept {
    return static_cast<T>(xsimd::fms(T(2.0), T_arg, T(hi)) * low);
}

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
C20CONSTEXPR void FuncEval<Func, N_compile_time, Iters_compile_time>::initialize_monomials(Func F,
                                                                                           const InputType *pts) {
    // 1) allocate
    Buffer<InputType, N_compile_time> grid{};
    if constexpr (N_compile_time == 0)
        grid.resize(n_terms);

    Buffer<OutputType, N_compile_time> samples{};
    if constexpr (N_compile_time == 0)
        samples.resize(n_terms);

    // 2) fill
    for (std::size_t k = 0; k < n_terms; ++k) {
        grid[k] = pts ? pts[k] : InputType(detail::cos((2.0 * InputType(k) + 1.0) * M_PI / (2.0 * n_terms)));
    }
    for (std::size_t i = 0; i < n_terms; ++i) {
        samples[i] = F(map_to_domain(grid[i]));
    }

    // 3) compute Newton → monomial
    auto newton = detail::bjorck_pereyra<N_compile_time, InputType, OutputType>(grid, samples);
    auto temp_monomial = detail::newton_to_monomial<N_compile_time, InputType, OutputType>(newton, grid);
    assert(temp_monomial.size() == monomials.size() && "size mismatch!");

    std::copy(temp_monomial.begin(), temp_monomial.end(), monomials.begin());

    // 4) optional refine
    refine(grid, samples);
}

template <class Func, std::size_t N_compile_time, std::size_t Iters_compile_time>
C20CONSTEXPR void
FuncEval<Func, N_compile_time, Iters_compile_time>::refine(const Buffer<InputType, N_compile_time> &x_cheb_,
                                                           const Buffer<OutputType, N_compile_time> &y_cheb_) {

    for (std::size_t pass = 0; pass < Iters_compile_time; ++pass) {
        // residuals
        Buffer<OutputType, N_compile_time> r_cheb;
        if constexpr (N_compile_time == 0) {
            r_cheb.resize(n_terms);
        }
        std::reverse(monomials.begin(), monomials.end());
        for (std::size_t i = 0; i < n_terms; ++i) {
            auto xi = x_cheb_[i];
            auto p_val = poly_eval::horner<N_compile_time>(xi, monomials.data(), monomials.size());
            r_cheb[i] = y_cheb_[i] - p_val;
        }
        std::reverse(monomials.begin(), monomials.end());
        // correction
        auto newton_r = detail::bjorck_pereyra<N_compile_time, InputType, OutputType>(x_cheb_, r_cheb);
        auto mono_r = detail::newton_to_monomial<N_compile_time, InputType, OutputType>(newton_r, x_cheb_);
        assert(mono_r.size() == monomials.size());

        for (std::size_t j = 0; j < monomials.size(); ++j) {
            monomials[j] += mono_r[j];
        }
    }
    std::reverse(monomials.begin(), monomials.end());
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
                std::cerr << "Warning: Requested epsilon " << eps
                          << " is less than machine epsilon for std::complex<float> ("
                          << std::numeric_limits<float>::epsilon() << "). Clamping.\n";
                eps = std::numeric_limits<float>::epsilon();
            }
        } else if constexpr (std::is_same_v<OutputType, std::complex<double>>) {
            if (eps < std::numeric_limits<double>::epsilon()) {
                std::cerr << "Warning: Requested epsilon " << eps
                          << " is less than machine epsilon for std::complex<double> ("
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
            const auto actual_val = F(pt);
            const auto poly_val = current_evaluator(pt);
            const auto current_abs_error = std::abs(1.0 - double(poly_val) / double(actual_val));
            if (current_abs_error > max_observed_error) {
                max_observed_error = current_abs_error;
            }
        }
        if (max_observed_error <= eps) {
            std::cout << "Converged: Found min degree N = " << n << " (Max Error: " << std::scientific
                      << std::setprecision(4) << max_observed_error << " <= Epsilon: " << eps << ")\n";
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
            double current_abs_error = std::abs(1.0 - poly_val / actual_val);
            if (current_abs_error > max_observed_error) {
                max_observed_error = current_abs_error;
            }
        }

        if (max_observed_error <= eps_val) {
            std::cout << "Converged: Found min degree N = " << n << " (Max Error: " << std::scientific
                      << std::setprecision(4) << max_observed_error << " <= Epsilon: " << eps_val << ")\n";
            return current_evaluator;
        }
    }

    std::cout << "Warning: Did not converge to epsilon " << std::scientific << std::setprecision(4) << eps_val
              << " within MaxN = " << MaxN_val << ". Returning FuncEval with degree " << MaxN_val << ".\n";
    return FuncEval<Func, 0, Iters_compile_time>(F, static_cast<int>(MaxN_val), a, b);
}
#endif

template <typename... EvalTypes> C20CONSTEXPR FuncEvalMany<EvalTypes...> make_func_eval(EvalTypes... evals) noexcept {
    return FuncEvalMany<std::decay_t<EvalTypes>...>(std::forward<EvalTypes>(evals)...);
}

} // namespace poly_eval