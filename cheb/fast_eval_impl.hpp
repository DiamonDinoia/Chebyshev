#pragma once

#include "macros.h"
#include "poly_eval.h"
#include "utils.h"

#include <cassert>
#include <iomanip>
#include <iostream>
#include <xsimd/xsimd.hpp>

namespace poly_eval {

namespace detail {} // namespace detail

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
        detail::unroll_loop<scalar_unroll>([&](const auto J) {
            constexpr auto i = decltype(J)::value;
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

// Constructor for static degree

template <class Func, std::size_t N_compile>
template <std::size_t C, typename>
constexpr FuncEvalND<Func, N_compile>::FuncEvalND(Func f, const InputType &a, const InputType &b)
    : func_{f}, degree_{static_cast<int>(N_compile)}, coeffs_flat_(), coeffs_md_{coeffs_flat_.data(), extents_t{}} {
    compute_scaling(a, b);
    initialize(static_cast<int>(N_compile));
}

// Constructor for dynamic degree
template <class Func, std::size_t N_compile>
template <std::size_t C, typename>
constexpr FuncEvalND<Func, N_compile>::FuncEvalND(Func f, int n, const InputType &a, const InputType &b)
    : func_{f}, degree_{n}, coeffs_flat_(storage_required(n)), coeffs_md_{coeffs_flat_.data(), make_ext(n)} {
    compute_scaling(a, b);
    initialize(n);
}

// Evaluate via Horner's method
template <class Func, std::size_t N_compile>
typename FuncEvalND<Func, N_compile>::OutputType FuncEvalND<Func, N_compile>::operator()(const InputType &x) const {
    return poly_eval::horner<N_compile, OutputType>(map_from_domain(x), coeffs_md_, degree_);
}

// coeff_impl
template <class Func, std::size_t N_compile>
template <typename IdxArray, std::size_t... I>
typename FuncEvalND<Func, N_compile>::Scalar &
FuncEvalND<Func, N_compile>::coeff_impl(const IdxArray &idx, std::size_t k, std::index_sequence<I...>) noexcept {
    return coeffs_md_(static_cast<std::size_t>(idx[I])..., k);
}

template <class Func, std::size_t N_compile>
template <class IdxArray>
[[nodiscard]] typename FuncEvalND<Func, N_compile>::Scalar &FuncEvalND<Func, N_compile>::coeff(const IdxArray &idx,
                                                                                               std::size_t k) noexcept {
    return coeff_impl<IdxArray>(idx, k, std::make_index_sequence<dim_>{});
}

template <class Func, std::size_t N_compile> auto FuncEvalND<Func, N_compile>::make_ext(int n) noexcept -> extents_t {
    if constexpr (is_static) {
        return detail::make_static_extents<N_compile, dim_, outDim_>(std::make_index_sequence<dim_>{});
    } else {
        return make_ext(n, std::make_index_sequence<dim_ + 1>{});
    }
}

template <class Func, std::size_t N_compile>
template <std::size_t... Is>
auto FuncEvalND<Func, N_compile>::make_ext(int n, std::index_sequence<Is...>) noexcept -> extents_t {
    return extents_t{(Is < dim_ ? static_cast<std::size_t>(n) : static_cast<std::size_t>(outDim_))...};
}

template <class Func, std::size_t N_compile>
constexpr std::size_t FuncEvalND<Func, N_compile>::storage_required(const int n) noexcept {
    auto ext = make_ext(n);
    auto mapping = typename mdspan_t::mapping_type{ext};
    return mapping.required_span_size();
}

template <class Func, std::size_t N_compile> constexpr void FuncEvalND<Func, N_compile>::initialize(int n) {
    Buffer<Scalar, N_compile> nodes{};
    if constexpr (!N_compile)
        nodes.resize(n);
    for (int k = 0; k < n; ++k)
        nodes[k] = detail::cos((2.0 * double(k) + 1.0) * M_PI / (2.0 * n));

    std::array<int, dim_> ext_idx{};
    ext_idx.fill(n);

    // sample f on Chebyshev grid
    for_each_index<dim_>(ext_idx, [&](const std::array<int, dim_> &idx) {
        InputType x_dom{};
        for (std::size_t d = 0; d < dim_; ++d)
            x_dom[d] = nodes[idx[d]];
        OutputType y = func_(map_to_domain(x_dom));
        for (std::size_t k = 0; k < outDim_; ++k)
            coeff(idx, k) = y[k];
    });

    // convert Newton → monomial along each axis
    Buffer<Scalar, N_compile> rhs{}, alpha{}, mono{};
    if constexpr (!N_compile) {
        rhs.resize(n);
        alpha.resize(n);
        mono.resize(n);
    }

    std::array<int, dim_> base_idx{};
    for (std::size_t axis = 0; axis < dim_; ++axis) {
        auto inner_ext = ext_idx;
        inner_ext[axis] = 1;
        for_each_index<dim_>(inner_ext, [&](const std::array<int, dim_> &base) {
            for (std::size_t k = 0; k < outDim_; ++k) {
                for (int i = 0; i < n; ++i) {
                    base_idx = base;
                    base_idx[axis] = i;
                    rhs[i] = coeff(base_idx, k);
                }
                alpha = detail::bjorck_pereyra<N_compile, Scalar, Scalar>(nodes, rhs);
                mono = detail::newton_to_monomial<N_compile, Scalar, Scalar>(alpha, nodes);
                for (int i = 0; i < n; ++i) {
                    base_idx = base;
                    base_idx[axis] = i;
                    coeff(base_idx, k) = mono[i];
                }
            }
        });
    }

    // reverse coefficient order
    for (std::size_t axis = 0; axis < dim_; ++axis) {
        auto inner_ext = ext_idx;
        inner_ext[axis] = 1;
        for_each_index<dim_>(inner_ext, [&](const std::array<int, dim_> &base) {
            for (std::size_t k = 0; k < outDim_; ++k) {
                int i = 0, j = n - 1;
                while (i < j) {
                    base_idx = base;
                    base_idx[axis] = i;
                    auto &a = coeff(base_idx, k);
                    base_idx[axis] = j;
                    auto &b = coeff(base_idx, k);
                    std::swap(a, b);
                    ++i;
                    --j;
                }
            }
        });
    }
}

template <class Func, std::size_t N_compile>
[[nodiscard]] constexpr typename FuncEvalND<Func, N_compile>::InputType
FuncEvalND<Func, N_compile>::map_to_domain(const InputType &t) const noexcept {
    InputType out{};
    for (std::size_t d = 0; d < dim_; ++d)
        out[d] = Scalar(0.5) * (t[d] / low_[d] + hi_[d]);
    return out;
}

template <class Func, std::size_t N_compile>
[[nodiscard]] constexpr typename FuncEvalND<Func, N_compile>::InputType
FuncEvalND<Func, N_compile>::map_from_domain(const InputType &x) const noexcept {
    InputType out{};
    for (std::size_t d = 0; d < dim_; ++d)
        out[d] = (Scalar(2) * x[d] - hi_[d]) * low_[d];
    return out;
}

template <class Func, std::size_t N_compile>
constexpr void FuncEvalND<Func, N_compile>::compute_scaling(const InputType &a, const InputType &b) noexcept {
    for (std::size_t d = 0; d < dim_; ++d) {
        low_[d] = Scalar(1) / (b[d] - a[d]);
        hi_[d] = b[d] + a[d];
    }
}

template <class Func, std::size_t N_compile>
template <std::size_t Rank, class F>
void FuncEvalND<Func, N_compile>::for_each_index(const std::array<int, Rank> &ext, F &&body) {
    std::array<int, Rank> idx{};
    while (true) {
        body(idx);
        for (std::size_t d = 0; d < Rank; ++d) {
            if (++idx[d] < ext[d])
                break;
            if (d == Rank - 1)
                return;
            idx[d] = 0;
        }
    }
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

    // Validate eps: cannot be less than machine precision for the input type
    if (eps < std::numeric_limits<InputType>::epsilon()) {
        std::cerr << "Warning: Requested epsilon " << eps << " is less than machine epsilon for InputType ("
                  << std::numeric_limits<InputType>::epsilon() << "). Clamping.\n";
        eps = std::numeric_limits<InputType>::epsilon();
    }

    std::vector<InputType> eval_points = detail::linspace(a, b, static_cast<int>(NumEvalPoints_val));

    for (int n = 1; n <= static_cast<int>(MaxN_val); ++n) {
        FuncEval<Func, 0, Iters_compile_time> current_evaluator(F, n, a, b);
        double max_observed_error = 0.0;
        for (const auto &pt : eval_points) {
            const auto actual_val = F(pt);
            const auto poly_val = current_evaluator(pt);
            const auto current_abs_error = detail::relative_error(poly_val, actual_val);
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
          class Func, typename InputType, typename>
NO_INLINE constexpr auto make_func_eval(Func F, InputType a, InputType b) {
    using OutputType = typename function_traits<Func>::result_type;

    static_assert(MaxN_val > 0, "Max polynomial degree must be positive.");
    static_assert(NumEvalPoints_val > 1, "Number of evaluation points must be greater than 1.");
    static_assert(eps_val >= std::numeric_limits<InputType>::epsilon(),
                  "Epsilon must be at most than machine epsilon for InputType.");

    std::vector<InputType> eval_points = detail::linspace(a, b, static_cast<int>(NumEvalPoints_val));

    for (int n = 1; n <= static_cast<int>(MaxN_val); ++n) {
        FuncEval<Func, 0, Iters_compile_time> current_evaluator(F, n, a, b);
        double max_observed_error = 0.0;

        for (const auto &pt : eval_points) {
            OutputType actual_val = F(pt);
            OutputType poly_val = current_evaluator(pt);
            double current_abs_error = detail::relative_error(poly_val, actual_val);
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

template <typename... EvalTypes, typename>
C20CONSTEXPR FuncEvalMany<EvalTypes...> make_func_eval(EvalTypes... evals) noexcept {
    return FuncEvalMany<std::decay_t<EvalTypes>...>(std::forward<EvalTypes>(evals)...);
}

// — static (compile‐time) degree
template <std::size_t N_compile_time, class Func, typename Fdec, typename InputType, typename>
NO_INLINE C20CONSTEXPR auto make_func_eval(Func &&F, InputType const &a, InputType const &b)
    -> FuncEvalND<Fdec, N_compile_time> {
    static_assert(N_compile_time > 0, "Degree must be > 0 for compile‑time overload");
    return FuncEvalND<Fdec, N_compile_time>(std::forward<Func>(F), a, b);
}

// — dynamic (run‐time) degree
template <class Func, typename Fdec, typename InputType, typename, typename>
auto make_func_eval(Func &&F, int n, InputType const &a, InputType const &b) noexcept -> FuncEvalND<Fdec, 0> {
    assert(n > 0 && "Degree must be positive for runtime overload");
    return FuncEvalND<Fdec, 0>(std::forward<Func>(F), n, a, b);
}

// — 3) C++17: Find minimal N ≤ MaxN_val to reach runtime eps
template <std::size_t MaxN_val, std::size_t NumEvalPoints_val, class Func, typename InputType, typename OutputType,
          typename>
NO_INLINE auto make_func_eval(Func &&F, double eps, InputType const &a, InputType const &b) -> FuncEvalND<Func, 0> {
    static_assert(MaxN_val > 0, "MaxN_val must be > 0");
    static_assert(NumEvalPoints_val > 1, "Need at least 2 eval points");
    // Clamp eps to machine-precision for InputType and warn if needed
    if (eps < std::numeric_limits<typename InputType::value_type>::epsilon()) {
        std::cerr << "Warning: Requested epsilon " << eps << " is less than machine epsilon for InputType ("
                  << std::numeric_limits<typename InputType::value_type>::epsilon() << "). Clamping.\n";
        eps = std::numeric_limits<typename InputType::value_type>::epsilon();
    }
    // generate NumEvalPoints_val linearly spaced points in [a,b]
    std::vector<InputType> eval_pts = detail::linspace(a, b, static_cast<int>(NumEvalPoints_val));
    double max_err = 0.0;
    for (int n = 2; n <= int(MaxN_val); ++n) {
        auto evaluator = FuncEvalND<Func, 0>(std::forward<Func>(F), n, a, b);
        max_err = 0.0;
        for (auto const &pt : eval_pts) {
            auto actual = F(pt);
            auto approx = evaluator(pt);
            max_err += detail::relative_l2_norm(actual, approx);
        }
        max_err /= eval_pts.size(); // average error over all points
        if (max_err <= eps) {
            std::cout << "Converged with N=" << n << " (max err=" << std::scientific << max_err << ")\n";
            return evaluator;
        }
    }

    std::cerr << "Warning: did not converge to eps=" << std::scientific << eps << " within N=" << MaxN_val
              << " (max err=" << std::scientific << max_err << ")\n";
    return FuncEvalND<Func, 0>(std::forward<Func>(F), static_cast<int>(MaxN_val), a, b);
}

#if __cplusplus >= 202002L
// — 4) C++20: eps_val as a template parameter
template <double eps, std::size_t MaxN_val, std::size_t NumEvalPoints_val, class Func, typename InputType,
          typename OutputType, typename>
NO_INLINE constexpr auto make_func_eval(Func &&F, InputType const &a, InputType const &b) -> FuncEvalND<Func, 0> {
    static_assert(MaxN_val > 0, "MaxN_val must be > 0");
    static_assert(NumEvalPoints_val > 1, "Need at least 2 eval points");
    static_assert(eps > std::numeric_limits<typename InputType::value_type>::epsilon(),
                  "eps must be greater than machine epsilon for InputType");

    std::vector<InputType> eval_pts = detail::linspace(a, b, static_cast<int>(NumEvalPoints_val));

    double max_err = 0.0;
    for (int n = 2; n <= int(MaxN_val); ++n) {
        auto evaluator = FuncEvalND<Func, 0>(std::forward<Func>(F), n, a, b);
        max_err = 0.0;
        for (auto const &pt : eval_pts) {
            auto actual = F(pt);
            auto approx = evaluator(pt);
            max_err += detail::relative_l2_norm(actual, approx);
        }
        max_err /= eval_pts.size(); // average error over all points
        if (max_err <= eps) {
            std::cout << "Converged with N=" << n << " (max err=" << std::scientific << max_err << ")\n";
            return evaluator;
        }
    }

    std::cerr << "Warning: did not converge to eps=" << std::scientific << eps << " within N=" << MaxN_val
              << " (max err=" << std::scientific << max_err << ")\n";
    return FuncEvalND<Func, 0>(std::forward<Func>(F), static_cast<int>(MaxN_val), a, b);
}
#endif

} // namespace poly_eval