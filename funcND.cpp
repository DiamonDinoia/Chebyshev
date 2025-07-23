#include <array>
#include <chrono>
#include <cmath>
#include <experimental/mdspan> // or <mdspan> in C++23
#include <functional>
#include <iostream>
#include <random>
#include <type_traits>
#include <utility> // for index_sequence
#include <vector>

#include "fast_eval.hpp" // for function_traits

namespace stdex = std::experimental;
using namespace poly_eval;

template <class Func, std::size_t N_compile = 0, std::size_t Iters_CT = 1> class FuncEvalND {
  public:
    using Input0 = typename poly_eval::function_traits<Func>::arg0_type;
    using InputType = std::remove_cvref_t<Input0>;
    using OutputType = typename poly_eval::function_traits<Func>::result_type;
    using Scalar = typename OutputType::value_type;

    static constexpr std::size_t dim_ = std::tuple_size_v<InputType>;
    static constexpr std::size_t outDim_ = std::tuple_size_v<OutputType>;

    /*----- constructors ----------------------------------------------------*/
    template <std::size_t C = N_compile, typename = std::enable_if_t<(C != 0)>>
    constexpr FuncEvalND(Func f, const InputType &a, const InputType &b) : func_(f), degree_(static_cast<int>(C)) {
        compute_scaling(a, b);
        initialize(static_cast<int>(C));
    }

    template <std::size_t C = N_compile, typename = std::enable_if_t<(C == 0)>>
    constexpr FuncEvalND(Func f, int n, const InputType &a, const InputType &b) : func_(f), degree_(n) {
        compute_scaling(a, b);
        initialize(n);
    }

    /*----- call ------------------------------------------------------------*/
    OutputType operator()(const InputType &x) const {
        return poly_eval::horner<N_compile, OutputType>(map_from_domain(x), coeffs_md_, degree_);
    }

  private:
    /*----- types -----------------------------------------------------------*/
    using extents_t = stdex::dextents<std::size_t, dim_ + 1>;
    using mdspan_t = stdex::mdspan<Scalar, extents_t, stdex::layout_left>;

    /*----- data ------------------------------------------------------------*/
    Func func_;
    const int degree_;
    InputType low_{}, hi_{};
    std::vector<Scalar> coeffs_flat_;
    mdspan_t coeffs_md_;

    /*----- helpers ---------------------------------------------------------*/
    template <std::size_t... Is> static constexpr extents_t make_ext(int n, std::index_sequence<Is...>) {
        return extents_t((Is < dim_ ? std::size_t(n) : std::size_t(outDim_))...);
    }
    static constexpr extents_t make_ext(int n) { return make_ext(n, std::make_index_sequence<dim_ + 1>{}); }

    /*----------- init: sample, invert Vandermonde, re-order ----------------*/
    void initialize(int n) {
        /* storage */
        coeffs_flat_.resize(mdspan_t(nullptr, make_ext(n)).mapping().required_span_size());
        coeffs_md_ = mdspan_t(coeffs_flat_.data(), make_ext(n));

        /* Chebyshev-like nodes */
        constexpr Scalar pi = 3.14159265358979323846;
        Buffer<Scalar, N_compile> nodes{};
        if constexpr (!N_compile)
            nodes.resize(n);
        for (int i = 0; i < n; ++i)
            nodes[i] = std::cos(pi * (i + .5) / Scalar(n));

        /* index → linear offset helper */
        std::array<int, dim_> idx{};
        auto offset = [&](std::size_t k) {
            return [&]<std::size_t... I>(std::index_sequence<I...>) {
                return coeffs_md_.mapping()(static_cast<std::size_t>(idx[I])..., k);
            }(std::make_index_sequence<dim_>{});
        };

        /* sample func on full tensor grid */
        auto sample = [&](auto &&self, std::size_t axis) -> void {
            if (axis == dim_) {
                InputType x{};
                for (std::size_t d = 0; d < dim_; ++d)
                    x[d] = nodes[idx[d]];
                OutputType y = func_(map_to_domain(x));
                for (std::size_t k = 0; k < outDim_; ++k)
                    coeffs_flat_[offset(k)] = y[k];
                return;
            }
            for (int i = 0; i < n; ++i) {
                idx[axis] = i;
                self(self, axis + 1);
            }
        };
        sample(sample, 0);

        /* invert each 1-D Vandermonde fibre with Björck–Pereyra */
        auto fibre = [&](auto &&self, std::size_t axis, std::size_t depth) -> void {
            if (depth == dim_) {
                Buffer<Scalar, N_compile> rhs{};
                if constexpr (!N_compile)
                    rhs.resize(n);
                for (std::size_t k = 0; k < outDim_; ++k) {
                    for (int i = 0; i < n; ++i) {
                        idx[axis] = i;
                        rhs[i] = coeffs_flat_[offset(k)];
                    }

                    const auto alpha = detail::bjorck_pereyra<N_compile, Scalar, Scalar>(nodes, rhs);
                    const auto mono = detail::newton_to_monomial<N_compile, Scalar, Scalar>(alpha, nodes);

                    for (int i = 0; i < n; ++i) {
                        idx[axis] = i;
                        coeffs_flat_[offset(k)] = mono[i];
                    }
                }
                return;
            }
            if (depth == axis) {
                self(self, axis, depth + 1);
                return;
            }
            for (int i = 0; i < n; ++i) {
                idx[depth] = i;
                self(self, axis, depth + 1);
            }
        };
        for (std::size_t a = 0; a < dim_; ++a)
            fibre(fibre, a, 0);

        /* reverse fibres (same as before) */
        auto reverse = [&](std::size_t axis) {
            std::array<int, dim_> id{};
            auto off = [&](std::size_t k, std::size_t d) {
                id[axis] = int(k);
                return [&]<std::size_t... I>(std::index_sequence<I...>) {
                    return coeffs_md_.mapping()(static_cast<std::size_t>(id[I])..., d);
                }(std::make_index_sequence<dim_>{});
            };
            auto rec = [&](auto &&self, std::size_t depth) -> void {
                if (depth == dim_) {
                    for (std::size_t d = 0; d < outDim_; ++d)
                        for (int k0 = 0, k1 = n - 1; k0 < k1; ++k0, --k1)
                            std::swap(coeffs_flat_[off(k0, d)], coeffs_flat_[off(k1, d)]);
                    return;
                }
                if (depth == axis) {
                    self(self, depth + 1);
                    return;
                }
                for (int i = 0; i < n; ++i) {
                    id[depth] = i;
                    self(self, depth + 1);
                }
            };
            rec(rec, 0);
        };
        for (std::size_t a = 0; a < dim_; ++a)
            reverse(a);
    }

    /* affine maps domain ↔ [-1,1]^d */
    InputType map_to_domain(const InputType &t) const noexcept {
        InputType out{};
        for (std::size_t d = 0; d < dim_; ++d)
            out[d] = Scalar(0.5) * (t[d] / low_[d] + hi_[d]);
        return out;
    }
    InputType map_from_domain(const InputType &x) const noexcept {
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
};

int main() {
    // --- choose dims here ---
    constexpr size_t DimIn = 4;
    constexpr size_t DimOut = 4;
    constexpr int N = 8;
    const int Ntest = 1000;

    using VecN = std::array<double, DimIn>;
    using OutM = std::array<double, DimOut>;

    // --- define your function fVec on R^DimIn → R^DimOut ---
    auto fScalar = [](VecN const &x) {
        double s = 0;
        for (double xi : x)
            s += std::pow(std::abs(std::sin(xi) + std::cos(xi)), 1.5) * std::cos(xi * xi);
        return s;
    };
    auto fVec = [&](VecN const &x) {
        OutM y{};
        for (size_t i = 0; i < DimOut; ++i) {
            auto xi = x;
            xi[i % DimIn] += static_cast<VecN::value_type>(i) / 2000.;
            y[i] = static_cast<VecN::value_type>(std::pow(fScalar(xi), i + 1));
        }
        return y;
    };

    // --- domain & degree ---
    VecN a{}, b{};
    a.fill(-1.0);
    b.fill(2.0);

    // --- build approximation ---
    auto t0 = std::chrono::high_resolution_clock::now();
    FuncEvalND<decltype(fVec), N> approx(fVec, a, b);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Init: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

    // --- RNG setup ---
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(a[0], b[0]);

    // --- benchmark analytical eval & sum ---
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
    double analytic_ms = std::chrono::duration<double, std::milli>(ta1 - ta0).count();
    std::cout << "Analytical eval over " << Ntest << " pts: " << analytic_ms << " ms, sumAnalytic=" << sumAnalytic
              << "\n";

    // --- benchmark polynomial eval & sum ---
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
    double poly_ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();
    std::cout << "Polynomial eval over " << Ntest << " pts: " << poly_ms << " ms, sumPoly=" << sumPoly << "\n";

    // --- compute relative L2 error on same points ---
    double err2 = 0.0, norm2 = 0.0;
    gen.seed(42);
    for (int i = 0; i < Ntest; ++i) {
        VecN x;
        for (auto &xi : x)
            xi = dist(gen);
        auto vE = fVec(x), vP = approx(x);
        for (size_t d = 0; d < DimOut; ++d) {
            double e = vE[d] - vP[d];
            err2 += e * e;
            norm2 += vE[d] * vE[d];
        }
    }
    std::cout << "Relative L2 error: " << std::sqrt(err2 / norm2) << "\n";

    return 0;
}