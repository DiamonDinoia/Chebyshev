// func_eval_nd.cpp – self‑contained demo of multidimensional Chebyshev
// approximation with cache‑friendly storage order and static mdspan extents
// when N_compile > 0
//
// Builds with GCC 13+, Clang 17+, MSVC 2022 17.9+
//     c++ -std=c++23 -O3 func_eval_nd.cpp -o func_eval_nd
//
// ---------------------------------------------------------------------------

#include <array>
#include <chrono>
#include <cmath>
#if __cpp_lib_mdspan >= 202207L
#include <mdspan>
namespace stdex = std;
#else
#include <experimental/mdspan>
namespace stdex = std::experimental;
#endif
#include <functional>
#include <iostream>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "fast_eval.hpp" // Buffer<>, bjorck_pereyra, newton_to_monomial, poly_eval::horner
using namespace poly_eval;

// --- Helper to create static extents when N_compile > 0 --------------------
template <std::size_t N_compile, std::size_t DimIn, std::size_t DimOut, std::size_t... Is>
constexpr auto make_static_extents(std::index_sequence<Is...>) {
    return stdex::extents<std::size_t, ((void)Is, N_compile)..., DimOut>{};
}

/*==========================================================================*
 *                  Generic N‑dimensional function approximator             *
 *==========================================================================*/
template <class Func, std::size_t N_compile = 0, std::size_t Iters_CT = 1> class FuncEvalND {
  public:
    using Input0 = typename function_traits<Func>::arg0_type;
    using InputType = std::remove_cvref_t<Input0>;
    using OutputType = typename function_traits<Func>::result_type;
    using Scalar = typename OutputType::value_type;

    static constexpr std::size_t dim_ = std::tuple_size_v<InputType>;
    static constexpr std::size_t outDim_ = std::tuple_size_v<OutputType>;
    static constexpr bool is_static = (N_compile > 0);

    using extents_t =
        std::conditional_t<is_static,
                           decltype(make_static_extents<N_compile, dim_, outDim_>(std::make_index_sequence<dim_>{})),
                           stdex::dextents<std::size_t, dim_ + 1>>;

    using mdspan_t = stdex::mdspan<Scalar, extents_t, stdex::layout_right>;

    /*---------------- Constructors ----------------*/
    template <std::size_t C = N_compile, typename = std::enable_if_t<(C != 0)>>
    constexpr FuncEvalND(Func f, const InputType &a, const InputType &b)
        : func_{f}, degree_{static_cast<int>(C)}, coeffs_flat_(storage_required(C)),
          coeffs_md_{coeffs_flat_.data(), extents_t{}} {
        compute_scaling(a, b);
        initialize(static_cast<int>(C));
    }

    template <std::size_t C = N_compile, typename = std::enable_if_t<(C == 0)>>
    constexpr FuncEvalND(Func f, int n, const InputType &a, const InputType &b)
        : func_{f}, degree_{n}, coeffs_flat_(storage_required(n)), coeffs_md_{coeffs_flat_.data(), make_ext(n)} {
        compute_scaling(a, b);
        initialize(n);
    }

    /*---------------- Evaluation operator ----------------*/
    OutputType operator()(const InputType &x) const {
        return poly_eval::horner<N_compile, OutputType>(map_from_domain(x), coeffs_md_, degree_);
    }

  private:
    /*-------------------- Data members -------------------*/
    Func func_;
    const int degree_;
    InputType low_{}, hi_{};
    std::vector<Scalar> coeffs_flat_; // owns the storage
    mdspan_t coeffs_md_;              // view over the storage

    /*======================================================
     *   Helper: turn an index array + output‑dimension idx
     *   into an lvalue reference into coeffs_md_
     *====================================================*/
    template <class IdxArray> [[nodiscard]] Scalar &coeff(const IdxArray &idx, std::size_t k) noexcept {
        return // expand idx[...] into mdspan::operator()
            [&]<std::size_t... I>(std::index_sequence<I...>) -> Scalar & {
                return coeffs_md_(static_cast<std::size_t>(idx[I])..., k);
            }(std::make_index_sequence<dim_>{});
    }

    /*-------------------- Extent helpers -----------------*/
    static extents_t make_ext(int n) {
        if constexpr (is_static) {
            return make_static_extents<N_compile, dim_, outDim_>(std::make_index_sequence<dim_>{});
        } else {
            return make_ext(n, std::make_index_sequence<dim_ + 1>{});
        }
    }
    template <std::size_t... Is> static extents_t make_ext(int n, std::index_sequence<Is...>) {
        return extents_t{(Is < dim_ ? std::size_t(n) : std::size_t(outDim_))...};
    }

    static constexpr std::size_t storage_required(int n) {
        auto ext = make_ext(n);
        auto mapping = typename mdspan_t::mapping_type{ext};
        return mapping.required_span_size();
    }

    /*-------------------- Initialisation -----------------*/
    void initialize(int n) {
        constexpr Scalar pi = 3.14159265358979323846;

        Buffer<Scalar, N_compile> nodes{};
        if constexpr (!N_compile)
            nodes.resize(n);
        for (int i = 0; i < n; ++i)
            nodes[i] = std::cos(pi * (i + 0.5) / Scalar(n));

        std::array<int, dim_> ext_idx{};
        ext_idx.fill(n);

        /*---- sample f on the Chebyshev grid ----*/
        for_each_index<dim_>(ext_idx, [&](const std::array<int, dim_> &idx) {
            InputType x_dom{};
            for (std::size_t d = 0; d < dim_; ++d)
                x_dom[d] = nodes[idx[d]];
            OutputType y = func_(map_to_domain(x_dom));
            for (std::size_t k = 0; k < outDim_; ++k)
                coeff(idx, k) = y[k];
        });

        /*---- convert along each axis: Newton → monomial ----*/
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

        /*---- reverse coefficient order in each axis ----*/
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

    InputType map_to_domain(const InputType &t) const noexcept {
        // t ∈ [‑1,1] → x ∈ [a,b]
        InputType out{};
        for (std::size_t d = 0; d < dim_; ++d)
            out[d] = Scalar(0.5) * (t[d] / low_[d] + hi_[d]);
        return out;
    }

    InputType map_from_domain(const InputType &x) const noexcept {
        // x ∈ [a,b] → t ∈ [‑1,1]
        InputType out{};
        for (std::size_t d = 0; d < dim_; ++d)
            out[d] = (Scalar(2) * x[d] - hi_[d]) * low_[d];
        return out;
    }

    void compute_scaling(const InputType &a, const InputType &b) noexcept {
        for (std::size_t d = 0; d < dim_; ++d) {
            low_[d] = Scalar(1) / (b[d] - a[d]);
            hi_[d] = b[d] + a[d];
        }
    }

    /*--------------------- Utilities ---------------------*/
    template <std::size_t Rank, class F> static void for_each_index(const std::array<int, Rank> &ext, F &&body) {
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
};

/*---------------------------------------------------------------------------*
 *                        Demo / micro‑benchmark                             *
 *---------------------------------------------------------------------------*/
int main() {
    constexpr std::size_t DimIn = 4;
    constexpr std::size_t DimOut = 4;
    constexpr int N = 8;    // polynomial degree
    const int Ntest = 1000; // evaluation points

    using VecN = std::array<double, DimIn>;
    using OutM = std::array<double, DimOut>;

    auto fScalar = [](const VecN &x) {
        double s = 0;
        for (double xi : x)
            s += std::pow(std::abs(std::sin(xi) + std::cos(xi)), 1.5) * std::cos(xi * xi);
        return s;
    };
    auto fVec = [&](const VecN &x) {
        OutM y{};
        for (std::size_t i = 0; i < DimOut; ++i) {
            auto xi = x;
            xi[i % DimIn] += float(i) / 2000.0f;
            y[i] = std::pow(fScalar(xi), static_cast<int>(i) + 1);
        }
        return y;
    };

    VecN a{}, b{};
    a.fill(-1.0f);
    b.fill(2.0f);

    auto t0 = std::chrono::high_resolution_clock::now();
    FuncEvalND<decltype(fVec), N> approx(fVec, a, b);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Init: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(a[0], b[0]);

    double sumAnalytic = 0.0;
    auto ta0 = std::chrono::high_resolution_clock::now();
    gen.seed(42);
    for (int i = 0; i < Ntest; ++i) {
        VecN x;
        for (auto &xi : x)
            xi = dist(gen);
        auto y = fVec(x);
        for (double v : y)
            sumAnalytic += v;
    }
    auto ta1 = std::chrono::high_resolution_clock::now();
    std::cout << "Analytical eval over " << Ntest
              << " pts: " << std::chrono::duration<double, std::milli>(ta1 - ta0).count()
              << " ms, sumAnalytic = " << sumAnalytic << '\n';

    double sumPoly = 0.0;
    auto tp0 = std::chrono::high_resolution_clock::now();
    gen.seed(42);
    for (int i = 0; i < Ntest; ++i) {
        VecN x;
        for (auto &xi : x)
            xi = dist(gen);
        auto y = approx(x);
        for (double v : y)
            sumPoly += v;
    }
    auto tp1 = std::chrono::high_resolution_clock::now();
    std::cout << "Polynomial eval over " << Ntest
              << " pts: " << std::chrono::duration<double, std::milli>(tp1 - tp0).count()
              << " ms, sumPoly = " << sumPoly << '\n';

    double err2 = 0.0, norm2 = 0.0;
    gen.seed(42);
    for (int i = 0; i < Ntest; ++i) {
        VecN x;
        for (auto &xi : x)
            xi = dist(gen);
        auto vE = fVec(x), vP = approx(x);
        for (std::size_t d = 0; d < DimOut; ++d) {
            double e = vE[d] - vP[d];
            err2 += e * e;
            norm2 += vE[d] * vE[d];
        }
    }
    std::cout << "Relative L2 error: " << std::sqrt(err2 / norm2) << '\n';
    return 0;
}
