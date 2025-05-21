#include <algorithm>
#include <vector>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <iomanip>
#include <cassert>

constexpr double PI = 3.14159265358979323846;

struct ChebData1D {
  int N;
  double a, b;
  std::vector<double> x; // Chebyshev–Lobatto nodes
  std::vector<double> w; // Barycentric weights

  ChebData1D(int n, double a_ = -1, double b_ = 1)
    : N(n), a(a_), b(b_), x(n), w(n) {
    for (int i = 0; i < N; ++i) {
      double theta = PI * i / (N - 1);
      x[i] = map_to_domain(std::cos(theta));
      w[i] = (i == 0 || i == N - 1) ? 0.5 : 1.0;
      w[i] *= (i % 2 == 0) ? 1.0 : -1.0;
    }
  }

  // Map from [-1, 1] to [a, b]
  template <class T>
  constexpr T map_to_domain(const T x) const {
    return 0.5 * ((b - a) * x + (b + a));
  }

  // Map from [a, b] to [-1, 1]
  constexpr double map_from_domain(double x) const {
    return (2.0 * x - (b + a)) / (b - a);
  }
};

template <class Func>
class Cheb1D {
public:
  Cheb1D(Func F, const ChebData1D &data)
    : N(data.N), low(data.b - data.a), hi(data.b + data.a), coeffs(N) {

    std::vector<double> fvals(N);
    for (int i = 0; i < N; ++i) {
      fvals[i] = F(data.x[i]);
    }

    for (int m = 0; m < N; ++m) {
      double sum = 0.5 * fvals[0];
      for (int k = 1; k < N - 1; ++k) {
        double theta = PI * k * m / (N - 1);
        sum += fvals[k] * std::cos(theta);
      }
      sum += 0.5 * fvals[N - 1] * std::cos(PI * m);
      coeffs[m] = 2.0 / (N - 1) * sum;
    }

    coeffs[0] *= 0.5;
    coeffs[N - 1] *= 0.5;
    std::reverse(coeffs.begin(), coeffs.end());
  }

  double operator()(const double pt) const {
    const double x = map_from_domain(pt);
    const double x2 = 2 * x;

    double c0 = coeffs[0];
    double c1 = coeffs[1];

    for (int i = 2; i < N; ++i) {
      const double tmp = c1;
      c1 = coeffs[i] - c0;
      c0 = c0 * x2 + tmp;
    }

    return c1 + c0 * x;
  }

private:
  const int N;
  double low, hi;
  std::vector<double> coeffs;

  constexpr double map_from_domain(double x) const {
    return (2.0 * x - hi) / low;
  }
};

template <class Func>
class BarCheb1D {
public:
  BarCheb1D(Func F, const ChebData1D &data)
    : N(data.N), a(data.a), b(data.b), x(data.x), w(data.w), fvals(N) {
    for (int i = 0; i < N; ++i) {
      fvals[i] = F(x[i]);
    }
  }

  double operator()(double pt) const {
    for (int i = 0; i < N; ++i) {
      if (pt == x[i]) { return fvals[i]; }
    }

    double num = 0, den = 0;
    for (int i = 0; i < N; ++i) {
      double diff = pt - x[i];
      double q = w[i] / diff;
      num += q * fvals[i];
      den += q;
    }

    return num / den;
  }

private:
  const int N;
  const double a, b;
  const std::vector<double> x, w;
  std::vector<double> fvals;
};

template <typename T, typename V>
void test(V &&f) {
  int n = 16;
  double a = -1.5, b = 1.5;

  ChebData1D data(n, a, b);
  T interpolator(f, data);

  std::cout << "Chebyshev interpolation test on random samples:\n";
  std::cout << "Function: f(x) = exp(x)+1, Domain: [" << a << ", " << b << "], Nodes: " << n << "\n\n";

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
  auto f = [](double x) { return std::exp(x) + 1; };
  test<Cheb1D<decltype(f)>>(f);
  std::cout << std::string(80, '-') << "\n\n\n";
  test<BarCheb1D<decltype(f)>>(f);
  return 0;
}
