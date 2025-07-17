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

    using CoeffType = OutputType::value_type;

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
        return horner<N_compile_time, OutputType>(x, coeffs_md_, degree_);
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
        const int samples = 2 * n;
        std::size_t terms = 1, gridSize = 1;
        for (std::size_t i = 0; i < dim_; ++i) {
            terms *= n;
            gridSize *= samples;
        }

        MatrixType V(gridSize, terms), powers(dim_, n), Y(gridSize, outDim_);

        std::vector<CoeffType> nodes(samples);
        for (int i = 0; i < samples; ++i)
            nodes[i] = static_cast<CoeffType>(std::cos((2.0 * i + 1.0) * M_PI / (2.0 * samples)));

        std::array<int, dim_> gridIdx{};

        for (std::size_t row = 0; row < gridSize; ++row) {
            InputType t{};
            for (std::size_t d = 0; d < dim_; ++d) {
                t[d] = nodes[gridIdx[d]];
                powers(d, 0) = CoeffType(1);
                for (int k = 1; k < n; ++k)
                    powers(d, k) = powers(d, k - 1) * t[d];
            }

            InputType x_phys = map_to_domain(t);
            auto fx = func_(x_phys);
            for (std::size_t d = 0; d < outDim_; ++d)
                Y(row, d) = fx[d];

            std::array<int, dim_> monoIdx{};
            for (std::size_t col = 0; col < terms; ++col) {
                CoeffType p = CoeffType(1);
                for (std::size_t d = 0; d < dim_; ++d)
                    p *= powers(d, monoIdx[d]);
                V(row, col) = p;

                // increment the multi‐index
                for (std::size_t d = 0; d < dim_; ++d) {
                    if (++monoIdx[d] == n)
                        monoIdx[d] = 0;
                    else
                        break;
                }
            }

            for (std::size_t d = 0; d < dim_; ++d) {
                if (++gridIdx[d] == samples)
                    gridIdx[d] = 0;
                else
                    break;
            }
        }

        Eigen::HouseholderQR<MatrixType> qr(V);
        MatrixType R_full = qr.matrixQR().template triangularView<Eigen::Upper>();
        MatrixType Q_full = qr.householderQ();

        const int cols = static_cast<int>(terms);
        auto Q = Q_full.leftCols(cols);
        auto R = R_full.topRows(cols);
        coeffs_flat_.resize(cols * outDim_);
        Eigen::Map<MatrixType> C(coeffs_flat_.data(), cols, outDim_);
        C = R.template triangularView<Eigen::Upper>().solve(Q.transpose() * Y);

        for (std::size_t pass = 0; pass < kItersCompileTime; ++pass) {
            auto r = Y - V * C;
            auto delta = R.template triangularView<Eigen::Upper>().solve(Q.transpose() * r).eval();
            C += delta;
        }

        // coeffs_flat_.assign(C.data(), C.data() + C.size());
        mapping_ = mapping_t{make_extents(n)};
        coeffs_md_ = mdspan_t{coeffs_flat_.data(), mapping_};
    }

    C20CONSTEXPR InputType map_to_domain(const InputType &t) const noexcept {
        InputType out{};
        for (std::size_t d = 0; d < dim_; ++d)
            out[d] = static_cast<typename InputType::value_type>(0.5 * (t[d] / low_[d] + hi_[d]));
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
    constexpr int N = 4;
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
        double r = fScalar(x);
        OutM y{};
        for (size_t i = 0; i < DimOut; ++i)
            y[i] = std::pow(r, i + 1);
        return y;
    };

    // --- domain & degree ---
    VecN a{}, b{};
    a.fill(-1.0);
    b.fill(1.0);

    // --- build approximation ---
    auto t0 = std::chrono::high_resolution_clock::now();
    FuncEvalND<decltype(fVec), N, 1> approx(fVec, a, b);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Init: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

    // --- RNG setup ---
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1, 1);

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