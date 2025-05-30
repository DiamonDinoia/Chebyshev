
#include <cheb1d.h>
#include <iostream>
#include <random>
#include <iomanip>
#include <cassert>


#include <xsimd/xsimd.hpp>

template <typename T, typename V>
void test(V &&f) {
  int n = 33; // Number of Chebyshev nodes (degree = n - 1)
  double a = -1.5, b = 1.5;

  T interpolator(f, n, a, b);

  std::cout << "Chebyshev interpolation test on random samples:\n";
  std::cout << "Function: f(x) = cos(x), Domain: [" << a << ", " << b << "], Nodes: " << n << "\n\n";

  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(a, b);

  std::cout << std::setprecision(6) << std::scientific;
  std::cout << "x"
      << std::setw(20) << "f(x)"
      << std::setw(20) << "Interp(x)"
      << std::setw(20) << "Rel. Error\n";
  std::cout << std::string(80, '-') << "\n";

  for (int i = 0; i < 15; ++i) {
    double x = dist(rng);
    double fx = f(x);
    double fx_interp = interpolator(x);
    double err = std::abs(1.0 - fx / fx_interp);
    std::cout << x << "\t" << fx << "\t" << fx_interp << "\t" << err << "\n";
  }
}

int main() {
  auto f = [](const double x) { return std::cos(x); };
  test<Cheb1D<decltype(f)>>(f);
  std::cout << std::string(80, '-') << "\n\n\n";
  test<BarCheb1D<decltype(f)>>(f);
  std::cout << std::string(80, '-') << "\n\n\n";
  test<Hor1D<decltype(f)>>(f);
  std::cout << std::string(80, '-') << "\n\n\n";
  test<FixedHor<decltype(f)>>(f);
  std::cout << std::string(80, '-') << "\n\n\n";
  test<Est1D<decltype(f)>>(f);
  std::cout << std::string(80, '-') << "\n\n\n";
  test<FixedEst<decltype(f)>>(f);
}

