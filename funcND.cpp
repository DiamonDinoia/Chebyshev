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
#include <functional>
#include <iostream>
#include <random>

#include "fast_eval.hpp" // Buffer<>, bjorck_pereyra, newton_to_monomial, poly_eval::horner
using namespace poly_eval;

/*---------------------------------------------------------------------------*
 *                        Demo / micro‑benchmark                             *
 *---------------------------------------------------------------------------*/
int main() {
    constexpr std::size_t DimIn = 4;
    constexpr std::size_t DimOut = 4;
    constexpr int N = 8;         // polynomial degree
    constexpr double eps = 1e-2; // error tolerance
    const int Ntest = 1000;      // evaluation points

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
    b.fill(0.5f);

    auto t0 = std::chrono::high_resolution_clock::now();
    const auto approx = poly_eval::make_func_eval<eps>(fVec, a, b);
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

    double relnorm2 = 0.0;
    gen.seed(42);
    for (int i = 0; i < Ntest; ++i) {
        VecN x;
        for (auto &xi : x)
            xi = dist(gen);
        auto vE = fVec(x), vP = approx(x);
        relnorm2 += detail::relative_l2_norm(vP, vE);
    }
    std::cout << "Mean Relative norm 2: " << relnorm2 / double(Ntest) << '\n';
    return 0;
}
