// func_eval_nd.cpp – self‑contained demo of multidimensional Chebyshev
// approximation with cache‑friendly storage order and a non‑recursive
// initialize().  Compile with:
//   g++ -std=c++20 -O3 -ffast-math -march=native func_eval_nd.cpp -o demo

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

#include "fast_eval.hpp"        // Buffer<>, bjorck_pereyra, newton_to_monomial,
                                 // poly_eval::horner, poly_eval::function_traits
using namespace poly_eval;

/* ------------------------------------------------------------------------
 * Generic helpers (header‑only, no deps)                                   */

/// Iterate over a rectangular integer domain ext[0]×…×ext[R-1]
template <std::size_t Rank, class F>
void for_each_index(const std::array<int, Rank>& ext, F&& body) {
    std::array<int, Rank> idx{};
    while (true) {
        body(idx);
        for (std::size_t d = 0; d < Rank; ++d) {
            if (++idx[d] < ext[d]) break;
            if (d == Rank - 1) return;  // finished all tuples
            idx[d] = 0;
        }
    }
}

/// Map logical index → flat offset for an mdspan mapping.
template <class Mapping,
          std::size_t Rminus1 = Mapping::extents_type::rank() - 1>
std::size_t offset(const Mapping& map,
                   const std::array<int, Rminus1>& idx,
                   std::size_t last = 0) {
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
        return map(static_cast<std::size_t>(idx[I])..., last);
    }(std::make_index_sequence<Rminus1>{});
}

/* ------------------------------------------------------------------------
 * Main approximation class                                                */

template <class Func, std::size_t N_compile = 0, std::size_t Iters_CT = 1>
class FuncEvalND {
  public:
    using Input0     = typename function_traits<Func>::arg0_type;
    using InputType  = std::remove_cvref_t<Input0>;
    using OutputType = typename function_traits<Func>::result_type;
    using Scalar     = typename OutputType::value_type;

    static constexpr std::size_t dim_    = std::tuple_size_v<InputType>;
    static constexpr std::size_t outDim_ = std::tuple_size_v<OutputType>;

    /* -------- constructors -------------------------------------------- */
    template <std::size_t C = N_compile,
              typename       = std::enable_if_t<(C != 0)>>
    constexpr FuncEvalND(Func f, const InputType& a, const InputType& b)
        : func_(f), degree_(static_cast<int>(C)) {
        compute_scaling(a, b);
        initialize(static_cast<int>(C));
    }

    template <std::size_t C = N_compile,
              typename       = std::enable_if_t<(C == 0)>>
    constexpr FuncEvalND(Func f, int n, const InputType& a, const InputType& b)
        : func_(f), degree_(n) {
        compute_scaling(a, b);
        initialize(n);
    }

    /* -------- call operator ------------------------------------------- */
    OutputType operator()(const InputType& x) const {
        return poly_eval::horner<N_compile, OutputType>(
            map_from_domain(x), coeffs_md_, degree_);
    }

  private:
    /* -------- types ---------------------------------------------------- */
    using extents_t = stdex::dextents<std::size_t, dim_ + 1>;
    using mdspan_t  = stdex::mdspan<Scalar, extents_t, stdex::layout_right>; // C‑order

    /* -------- data ----------------------------------------------------- */
    Func              func_;
    const int         degree_;
    InputType         low_{}, hi_{};
    std::vector<Scalar> coeffs_flat_;
    mdspan_t            coeffs_md_;

    /* -------- helpers -------------------------------------------------- */
    template <std::size_t... Is>
    static constexpr extents_t make_ext(int n, std::index_sequence<Is...>) {
        return extents_t((Is < dim_ ? std::size_t(n) : std::size_t(outDim_))...);
    }
    static constexpr extents_t make_ext(int n) {
        return make_ext(n, std::make_index_sequence<dim_ + 1>{});
    }

    /* -------- initialize(): no recursion, cache‑friendly -------------- */
    void initialize(int n) {
        // 0. allocate contiguous storage
        coeffs_flat_.resize(mdspan_t(nullptr, make_ext(n)).mapping().required_span_size());
        coeffs_md_ = mdspan_t(coeffs_flat_.data(), make_ext(n));

        // 1. Chebyshev nodes on [-1,1]
        constexpr Scalar pi = 3.14159265358979323846;
        Buffer<Scalar, N_compile> nodes{};
        if constexpr (!N_compile) nodes.resize(n);
        for (int i = 0; i < n; ++i)
            nodes[i] = std::cos(pi * (i + .5) / Scalar(n));

        // 2. sample user function on full tensor grid
        std::array<int, dim_> ext; ext.fill(n);
        for_each_index<dim_>(ext, [&](const std::array<int, dim_>& idx) {
            InputType x_dom{};
            for (std::size_t d = 0; d < dim_; ++d)
                x_dom[d] = nodes[idx[d]];
            OutputType y = func_(map_to_domain(x_dom));
            for (std::size_t k = 0; k < outDim_; ++k)
                coeffs_flat_[offset(coeffs_md_.mapping(), idx, k)] = y[k];
        });

        // 3. invert Vandermonde fibre‑by‑fibre with Bjorck–Pereyra & Newton
        Buffer<Scalar, N_compile> rhs{}, alpha{}, mono{};
        if constexpr (!N_compile) { rhs.resize(n); alpha.resize(n); mono.resize(n); }

        std::array<int, dim_> idx{};
        for (std::size_t axis = 0; axis < dim_; ++axis) {
            std::array<int, dim_> inner_ext = ext; inner_ext[axis] = 1;
            for_each_index<dim_>(inner_ext, [&](const std::array<int, dim_>& base) {
                for (std::size_t k = 0; k < outDim_; ++k) {
                    // gather RHS along <axis>
                    for (int i = 0; i < n; ++i) {
                        idx = base; idx[axis] = i;
                        rhs[i] = coeffs_flat_[offset(coeffs_md_.mapping(), idx, k)];
                    }
                    // solve and convert
                    alpha = detail::bjorck_pereyra<N_compile, Scalar, Scalar>(nodes, rhs);
                    mono  = detail::newton_to_monomial<N_compile, Scalar, Scalar>(alpha, nodes);
                    // scatter back
                    for (int i = 0; i < n; ++i) {
                        idx = base; idx[axis] = i;
                        coeffs_flat_[offset(coeffs_md_.mapping(), idx, k)] = mono[i];
                    }
                }
            });
        }

        // 4. reverse each polynomial axis (Chebyshev → ascending monomial)
        for (std::size_t axis = 0; axis < dim_; ++axis) {
            std::array<int, dim_> inner_ext = ext; inner_ext[axis] = 1;
            for_each_index<dim_>(inner_ext, [&](const std::array<int, dim_>& base) {
                for (std::size_t k = 0; k < outDim_; ++k) {
                    int i = 0, j = n - 1;
                    while (i < j) {
                        idx = base; idx[axis] = i;
                        auto off_i = offset(coeffs_md_.mapping(), idx, k);
                        idx[axis] = j;
                        auto off_j = offset(coeffs_md_.mapping(), idx, k);
                        std::swap(coeffs_flat_[off_i], coeffs_flat_[off_j]);
                        ++i; --j;
                    }
                }
            });
        }
    }

    /* -------- affine maps domain ↔ [-1,1]^d --------------------------- */
    InputType map_to_domain(const InputType& t) const noexcept {
        InputType out{};
        for (std::size_t d = 0; d < dim_; ++d)
            out[d] = Scalar(0.5) * (t[d] / low_[d] + hi_[d]);
        return out;
    }
    InputType map_from_domain(const InputType& x) const noexcept {
        InputType out{};
        for (std::size_t d = 0; d < dim_; ++d)
            out[d] = (Scalar(2) * x[d] - hi_[d]) * low_[d];
        return out;
    }
    void compute_scaling(const InputType& a, const InputType& b) noexcept {
        for (std::size_t d = 0; d < dim_; ++d) {
            low_[d] = Scalar(1) / (b[d] - a[d]);
            hi_[d]  = b[d] + a[d];
        }
    }
};

/* ------------------------------------------------------------------------
 * Demo / benchmark                                                        */
int main() {
    constexpr std::size_t DimIn  = 4;
    constexpr std::size_t DimOut = 4;
    constexpr int         N      = 8;      // polynomial degree
    const int             Ntest  = 1000;   // evaluation points

    using VecN = std::array<double, DimIn>;
    using OutM = std::array<double, DimOut>;

    // scalar helper
    auto fScalar = [](const VecN& x) {
        double s = 0;
        for (double xi : x)
            s += std::pow(std::abs(std::sin(xi) + std::cos(xi)), 1.5) * std::cos(xi * xi);
        return s;
    };
    // vector‑valued target function
    auto fVec = [&](const VecN& x) {
        OutM y{};
        for (std::size_t i = 0; i < DimOut; ++i) {
            auto xi = x;
            xi[i % DimIn] += static_cast<VecN::value_type>(i) / 2000.0;
            y[i] = std::pow(fScalar(xi), static_cast<int>(i) + 1);
        }
        return y;
    };

    // domain
    VecN a{}, b{}; a.fill(-1.0); b.fill(2.0);

    // build approximation
    auto t0 = std::chrono::high_resolution_clock::now();
    FuncEvalND<decltype(fVec), N> approx(fVec, a, b);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Init: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

    // RNG setup
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(a[0], b[0]);

    // analytical eval benchmark
    double sumAnalytic = 0.0;
    auto ta0 = std::chrono::high_resolution_clock::now();
    gen.seed(42);
    for (int i = 0; i < Ntest; ++i) {
        VecN x; for (auto& xi : x) xi = dist(gen);
        auto y = fVec(x);
        for (double v : y) sumAnalytic += v;
    }
    auto ta1 = std::chrono::high_resolution_clock::now();
    std::cout << "Analytical eval over " << Ntest << " pts: "
              << std::chrono::duration<double, std::milli>(ta1 - ta0).count()
              << " ms, sumAnalytic=" << sumAnalytic << '\n';

    // polynomial eval benchmark
    double sumPoly = 0.0;
    auto tp0 = std::chrono::high_resolution_clock::now();
    gen.seed(42);
    for (int i = 0; i < Ntest; ++i) {
        VecN x; for (auto& xi : x) xi = dist(gen);
        auto y = approx(x);
        for (double v : y) sumPoly += v;
    }
    auto tp1 = std::chrono::high_resolution_clock::now();
    std::cout << "Polynomial eval over " << Ntest << " pts: "
              << std::chrono::duration<double, std::milli>(tp1 - tp0).count()
              << " ms, sumPoly=" << sumPoly << '\n';

    // relative L2 error
    double err2 = 0.0, norm2 = 0.0;
    gen.seed(42);
    for (int i = 0; i < Ntest; ++i) {
        VecN x; for (auto& xi : x) xi = dist(gen);
        auto vE = fVec(x), vP = approx(x);
        for (std::size_t d = 0; d < DimOut; ++d) {
            double e = vE[d] - vP[d];
            err2  += e * e;
            norm2 += vE[d] * vE[d];
        }
    }
    std::cout << "Relative L2 error: " << std::sqrt(err2 / norm2) << '\n';

    return 0;
}
