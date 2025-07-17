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
    constexpr FuncEvalND(Func f, const InputType &a, const InputType &b)
        : func_(f), degree_(static_cast<int>(C)), low_(a), hi_(b) {
        initialize(static_cast<int>(C), a, b);
    }

    // Run‑time degree constructor
    template <std::size_t C = N_compile_time, typename = std::enable_if_t<(C == 0)>>
    constexpr FuncEvalND(Func f, int n, const InputType &a, const InputType &b)
        : func_(f), degree_(n), low_(a), hi_(b) {
        initialize(n, a, b);
    }

    [[gnu::always_inline]]
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

    mapping_t mapping_;
    mdspan_t coeffs_md_;

    /* ---------- helpers ---------- */
    template <std::size_t... Is> static constexpr extents_t make_extents(int n, std::index_sequence<Is...>) {
        return extents_t{((void)Is, std::size_t(Is < dim_ ? n : outDim_))...};
    }
    static constexpr extents_t make_extents(int n) { return make_extents(n, std::make_index_sequence<dim_ + 1>{}); }

    void initialize(int n, const InputType &a, const InputType &b) {
        const int samples = 2 * n;
        std::size_t gridSize = 1;
        for (std::size_t i = 0; i < dim_; ++i)
            gridSize *= samples;

        std::size_t terms = 1;
        for (std::size_t i = 0; i < dim_; ++i)
            terms *= n;

        Eigen::MatrixXd V(gridSize, terms);
        Eigen::MatrixXd Y(gridSize, outDim_);

        /* Chebyshev nodes on [-1,1] */
        std::vector<double> nodes(samples);
        for (int i = 0; i < samples; ++i)
            nodes[i] = std::cos((2.0 * i + 1.0) * M_PI / (2.0 * samples));

        /* Fill Vandermonde V and samples Y */
        for (std::size_t idx = 0; idx < gridSize; ++idx) {
            std::size_t tmp = idx;
            InputType x;
            for (std::size_t d = 0; d < dim_; ++d) {
                int id = static_cast<int>(tmp % samples);
                tmp /= samples;
                double t = nodes[id];
                x[d] = 0.5 * (a[d] + b[d]) + 0.5 * (b[d] - a[d]) * t;
            }
            auto fx = func_(x);
            for (std::size_t d = 0; d < outDim_; ++d)
                Y(idx, d) = fx[d];

            for (std::size_t mon = 0; mon < terms; ++mon) {
                std::size_t code = mon;
                double pval = 1.0;
                for (std::size_t d = 0; d < dim_; ++d) {
                    int p = static_cast<int>(code % n);
                    code /= n;
                    pval *= std::pow(x[d], p);
                }
                V(idx, mon) = pval;
            }
        }

        using namespace Eigen;
        HouseholderQR<MatrixXd> qr(V);
        MatrixXd R_full = qr.matrixQR().triangularView<Upper>();
        MatrixXd Q_full = qr.householderQ();

        const int cols = static_cast<int>(terms);
        MatrixXd Q = Q_full.leftCols(cols);
        MatrixXd R = R_full.topRows(cols);

        MatrixXd C = R.triangularView<Upper>().solve(Q.transpose() * Y);

        /* iterative refinement controlled by compile-time constant */
        for (std::size_t pass = 0; pass < kItersCompileTime; ++pass) {
            MatrixXd r = Y - V * C;
            MatrixXd delta = R.triangularView<Upper>().solve(Q.transpose() * r);
            C += delta;
        }

        /* Flatten & wrap */
        coeffs_flat_.assign(C.data(), C.data() + C.size());
        mapping_ = mapping_t{make_extents(n)};
        coeffs_md_ = mdspan_t{coeffs_flat_.data(), mapping_};
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