#include <Eigen/Dense>
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

template <class Func, std::size_t N_compile_time = 0, std::size_t Iters_compile_time = 1> class FuncEvalND {
  public:
    using InputArg0 = typename function_traits<Func>::arg0_type;
    using InputType = std::remove_cvref_t<InputArg0>;
    using OutputType = typename function_traits<Func>::result_type;
    using CoeffType = typename OutputType::value_type;

    static constexpr std::size_t dim_ = std::tuple_size_v<InputType>;
    static constexpr std::size_t outDim_ = std::tuple_size_v<OutputType>;

    static constexpr std::size_t kDegreeCompileTime = N_compile_time;
    static constexpr std::size_t kItersCompileTime = Iters_compile_time;

    // Compile‑time degree constructor
    template <std::size_t C = N_compile_time, typename = std::enable_if_t<(C != 0)>>
    constexpr FuncEvalND(Func f, const InputType &a, const InputType &b) : func_(f), degree_(static_cast<int>(C)) {
        compute_scaling(a, b);
        initialize(static_cast<int>(C));
    }

    // Run‑time degree constructor
    template <std::size_t C = N_compile_time, typename = std::enable_if_t<(C == 0)>>
    constexpr FuncEvalND(Func f, int n, const InputType &a, const InputType &b) : func_(f), degree_(n) {
        compute_scaling(a, b);
        initialize(n);
    }

    OutputType operator()(const InputType &x) const {
        return horner<N_compile_time, OutputType>(map_from_domain(x), coeffs_md_, degree_);
    }

  private:
    /* ---------- data ---------- */
    Func func_;
    const int degree_;
    InputType low_, hi_;

    std::vector<CoeffType> coeffs_flat_;
    using extents_t = stdex::dextents<std::size_t, dim_ + 1>;
    using mdspan_t = stdex::mdspan<CoeffType, extents_t, stdex::layout_left>;
    using mapping_t = typename mdspan_t::mapping_type;
    using MatrixType = Eigen::Matrix<CoeffType, Eigen::Dynamic, Eigen::Dynamic>;

    mapping_t mapping_;
    mdspan_t coeffs_md_;

    /* ---------- helpers ---------- */
    template <std::size_t... Is> static constexpr extents_t make_extents(int n, std::index_sequence<Is...>) {
        return extents_t{((void)Is, std::size_t(Is < dim_ ? n : outDim_))...};
    }
    static constexpr extents_t make_extents(int n) { return make_extents(n, std::make_index_sequence<dim_ + 1>{}); }

    void initialize(int n) {
        /* ---------- storage ---------- */
        extents_t ext = make_extents(n); // (n,…,n,outDim_)
        coeffs_flat_.resize(mdspan_t(nullptr, ext).mapping().required_span_size());
        coeffs_md_ = mdspan_t(coeffs_flat_.data(), ext);
        mapping_ = coeffs_md_.mapping(); // keep a copy

        using Scalar = CoeffType; // float in your test

        /* ---------- Chebyshev-like nodes and Vandermonde ---------- */
        constexpr Scalar pi = 3.14159265358979323846f;
        std::vector<Scalar> nodes(n);
        for (int i = 0; i < n; ++i)
            nodes[i] = std::cos(pi * (i + 0.5f) / Scalar(n));

        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> V(n, n);
        for (int i = 0; i < n; ++i) {
            V(i, 0) = Scalar(1);
            for (int j = 1; j < n; ++j)
                V(i, j) = V(i, j - 1) * nodes[i];
        }
        Eigen::PartialPivLU<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> V_lu(V);

        /* ---------- helpers ---------- */
        std::array<int, dim_> idx{};

        auto offset = [&](std::size_t k) {
            return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return mapping_(static_cast<std::size_t>(idx[Is])..., k); // dim_ indices + output axis
            }(std::make_index_sequence<dim_>{});
        };

        /* ---------- sample f on full grid ---------- */
        auto sample_rec = [&](auto &&self, std::size_t axis) -> void {
            if (axis == dim_) {
                InputType x{};
                for (std::size_t d = 0; d < dim_; ++d)
                    x[d] = nodes[idx[d]]; // map to Chebyshev-like nodes
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
        sample_rec(sample_rec, 0);

        /* ---------- separable 1-D solves ---------- */
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> rhs(n), sol(n);

        auto fibre_rec = [&](auto &&self, std::size_t axis, std::size_t depth) -> void {
            if (depth == dim_) { // fixed all but 'axis'
                for (std::size_t k = 0; k < outDim_; ++k) {
                    for (int i = 0; i < n; ++i) {
                        idx[axis] = i;
                        rhs(i) = coeffs_flat_[offset(k)];
                    }
                    sol = V_lu.solve(rhs);
                    for (int i = 0; i < n; ++i) {
                        idx[axis] = i;
                        coeffs_flat_[offset(k)] = sol(i);
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
            fibre_rec(fibre_rec, a, 0);

        auto reverse_fibres = [&](std::size_t axis) {
            std::array<int, dim_> id{};
            auto off = [&](std::size_t k, std::size_t d) {
                id[axis] = int(k);
                /* build linear offset for id[0] … id[dim_-1] and output d */
                return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                    return mapping_(static_cast<std::size_t>(id[Is])..., d);
                }(std::make_index_sequence<dim_>{});
            };

            /* recurse over all indices except the chosen axis */
            auto rec = [&](auto &&self, std::size_t depth) -> void {
                if (depth == dim_) // all spatial axes fixed
                {
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
            reverse_fibres(a);
    }

    C20CONSTEXPR InputType map_to_domain(const InputType &t) const noexcept {
        InputType out{};
        for (std::size_t d = 0; d < dim_; ++d)
            out[d] = static_cast<typename InputType::value_type>(static_cast<typename InputType::value_type>(0.5) *
                                                                 (t[d] / low_[d] + hi_[d]));
        return out;
    }

    ALWAYS_INLINE C20CONSTEXPR InputType map_from_domain(const InputType &x) const noexcept {
        InputType out{};
        for (std::size_t d = 0; d < dim_; ++d)
            out[d] = static_cast<typename InputType::value_type>((typename InputType::value_type(2) * x[d] - hi_[d]) *
                                                                 low_[d]);
        return out;
    }

    void compute_scaling(const InputType &a, const InputType &b) noexcept {
        for (std::size_t d = 0; d < dim_; ++d) {
            low_[d] = typename InputType::value_type(1) / (b[d] - a[d]);
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

    using VecN = std::array<float, DimIn>;
    using OutM = std::array<float, DimOut>;

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
            y[i] = static_cast<VecN::value_type>(std::pow(fScalar(xi), i+1));
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