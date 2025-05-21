#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>


constexpr double PI = 3.14159265358979323846;

template <typename Real = double>
struct ChebGrid1D {
  int N;
  double a, b;
  std::vector<Real> x; // Chebyshev–Lobatto nodes
  std::vector<Real> w; // Barycentric weights

  ChebGrid1D(int n, double a_ = -1, double b_ = 1)
    : N(n), a(a_), b(b_), x(n), w(n) {
    for (int i = 0; i < N; ++i) {
      double theta = PI * i / (N - 1);
      x[i] = 0.5 * ((b - a) * std::cos(theta) + (b + a));

      w[i] = (i == 0 || i == N - 1) ? 0.5 : 1.0;
      w[i] *= (i % 2 == 0) ? 1.0 : -1.0;
    }
  }

  // Normalize to [-1, 1]
  Real to_cheb_domain(Real x_phys) const {
    return (2.0 * x_phys - (b + a)) / (b - a);
  }

  // Map to physical domain
  Real to_physical_domain(Real x_cheb) const {
    return 0.5 * ((b - a) * x_cheb + (b + a));
  }
};

template <class Func>
class Cheb1D {
public:
  Cheb1D(Func F, const ChebGrid1D<double>& grid)
    : N(grid.N), grid(grid), coeffs(N) {

    std::vector<double> fvals(N);
    for (int i = 0; i < N; ++i)
      fvals[i] = F(grid.x[i]);

    for (int m = 0; m < N; ++m) {
      double sum = 0.5 * fvals[0];
      for (int k = 1; k < N - 1; ++k) {
        sum += fvals[k] * std::cos(PI * k * m / (N - 1));
      }
      sum += 0.5 * fvals[N - 1] * std::cos(PI * m);
      coeffs[m] = 2.0 / (N - 1) * sum;
    }

    coeffs[0] *= 0.5;
    coeffs[N - 1] *= 0.5;
    std::reverse(coeffs.begin(), coeffs.end());
  }

  double operator()(double pt) const {
    double x = grid.to_cheb_domain(pt);
    double x2 = 2 * x;

    double c0 = coeffs[0];
    double c1 = coeffs[1];

    for (int i = 2; i < N; ++i) {
      double tmp = c1;
      c1 = coeffs[i] - c0;
      c0 = c0 * x2 + tmp;
    }

    return c1 + c0 * x;
  }

private:
  const int N;
  const ChebGrid1D<double>& grid;
  std::vector<double> coeffs;
};

template <class Func>
class BarCheb1D {
public:
  BarCheb1D(Func F, const ChebGrid1D<double>& grid)
    : N(grid.N), grid(grid), fvals(N) {
    for (int i = 0; i < N; ++i)
      fvals[i] = F(grid.x[i]);
  }

  double operator()(double pt) const {
    for (int i = 0; i < N; ++i) {
      if (pt == grid.x[i]) return fvals[i];
    }

    double num = 0, den = 0;
    for (int i = 0; i < N; ++i) {
      double diff = pt - grid.x[i];
      double q = grid.w[i] / diff;
      num += q * fvals[i];
      den += q;
    }

    return num / den;
  }

private:
  const int N;
  const ChebGrid1D<double>& grid;
  std::vector<double> fvals;
};


template <typename T, typename V>
void test(V &&f) {
  int n = 16;
  double a = -1.5, b = 1.5;
  ChebGrid1D<double> grid(n, a, b);

  T interpolator(f, grid);

  std::cout << "Chebyshev interpolation test on random samples:\n";
  std::cout << "Function: f(x) = sin(x)+1, Domain: [" << a << ", " << b << "], Nodes: " << n << "\n\n";

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
  auto f = [](double x) { return std::sin(x) + 1; };
  test<Cheb1D<decltype(f)>>(f);
  std::cout << std::string(80, '-') << "\n\n\n";
  test<BarCheb1D<decltype(f)>>(f);
  return 0;
}
