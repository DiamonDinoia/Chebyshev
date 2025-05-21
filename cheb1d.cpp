#include <algorithm>
#include <vector>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <iomanip>
#include <cassert>


constexpr double PI = 3.14159265358979323846;


template <class Func>
class Cheb1D {
public:
  Cheb1D(Func F, const int degree, const double a = -1, const double b = 1)
    : N(degree + 1), low(b - a), hi(b + a), coeffs(N) {

    std::vector<double> fvals(N); // N = degree + 1 points

    for (int k = 0; k < N; ++k) {
      double xk = std::cos(PI * k / degree); // extrema nodes
      double x_mapped = map_to_domain(xk);
      fvals[k] = F(x_mapped);
    }

    // Compute Chebyshev coefficients using DCT-I-style formula
    for (int n = 0; n < N; ++n) {
      double sum = .5 * fvals[0];
      for (int k = 1; k < degree; ++k) {
        double theta = PI * k * n / degree;
        sum += fvals[k] * std::cos(theta);
      }
      sum += 0.5 * fvals[degree] * std::cos(PI * n);
      coeffs[n] = 2.0 / degree * sum;
    }

    coeffs[0] *= 0.5;
    coeffs[degree] *= 0.5;
    // reverse coeffs
    std::reverse(coeffs.begin(), coeffs.end());
  }

  // Evaluate using Clenshaw's algorithm (numerically stable)
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

  // Map from [-1, 1] to [a, b]
  template <class T>
  constexpr auto map_to_domain(const T x) const {
    return 0.5 * (low * x + hi);
  }

  // Map from [a, b] to [-1, 1]
  constexpr double map_from_domain(double x) const {
    return (2.0 * x - hi) / low;
  }
};

template <class Func>
class BarCheb1D {
public:
  BarCheb1D(Func F, const int degree, const double a = -1, const double b = 1)
    : N(degree + 1), a(a), b(b), x(N), w(N), fvals(N) {
    for (int i = 0; i < N; ++i) {
      double theta = PI * i / (N - 1); // Lobatto points
      x[i] = map_to_domain(std::cos(theta));

      // Barycentric weights for Chebyshev–Lobatto nodes
      w[i] = (i == 0 || i == N - 1) ? 0.5 : 1.0;
      w[i] *= (i % 2 == 0) ? 1.0 : -1.0;

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
  std::vector<double> x, w, fvals;

  // Map from [-1, 1] to [a, b]
  template <class T>
  constexpr auto map_to_domain(const T x) const {
    return 0.5 * ((b - a) * x + (b + a));
  }

  // Map from [a, b] to [-1, 1]
  constexpr double map_from_domain(double x) const {
    return (2.0 * x - (b + a)) / (b - a);
  }
};

template <typename T, typename V> void test(V &&f) {

  int degree = 15;
  double a = -1.5, b = 1.5;

  T interpolator(f, degree, a, b);

  std::cout << "Chebyshev interpolation test on random samples:\n";
  std::cout << "Function: f(x) = exp(x)+1, Domain: [" << a << ", " << b << "], Degree: " << degree << "\n\n";

  // Set up RNG
  std::mt19937 rng(42); // Fixed seed for reproducibility
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


// ----- Main Test with Random Evaluation -----
int main() {
  auto f = [](double x) {
    return std::exp(x) + 1;
  };
  test<Cheb1D<decltype(f)>>(f);
  std::cout << std::string(80, '-') << "\n";
  std::cout << "\n\n\n";
  test<BarCheb1D<decltype(f)>>(f);
  return 0;
}