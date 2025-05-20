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
    : N(degree), a(a), b(b), coeffs(degree + 1) {
    // Use Chebyshev extrema nodes: x_k = cos(pi * k / N), for k = 0..N
    std::vector<double> fvals(N + 1);
    for (int k = 0; k <= N; ++k) {
      double xk = std::cos(PI * k / N);
      double x_mapped = map_to_domain(xk);
      fvals[k] = F(x_mapped);
    }

    // Compute Chebyshev coefficients using exact DCT-I formula (manual)
    for (int n = 0; n <= N; ++n) {
      double sum = 0.0;
      for (int k = 0; k <= N; ++k) {
        double theta = PI * k * n / N;
        double weight = 1.0;
        if (k == 0 || k == N)
          weight = 0.5; // edge weights
        sum += weight * fvals[k] * std::cos(theta);
      }
      coeffs[n] = (2.0 / N) * sum;
    }
    coeffs[0] *= 0.5; // adjust a_0
    coeffs[N] *= 0.5; // adjust a_N
  }

  // Evaluate using Clenshaw's algorithm (numerically stable)
  double operator()(double pt) const {
    double x = map_from_domain(pt); // map pt from [a, b] to [-1, 1]
    double b_kp1 = 0.0, b_kp2 = 0.0;
    for (int j = N; j >= 1; --j) {
      double b_k = 2.0 * x * b_kp1 - b_kp2 + coeffs[j];
      b_kp2 = b_kp1;
      b_kp1 = b_k;
    }
    return x * b_kp1 - b_kp2 + coeffs[0];
  }

private:
  int N;
  double a, b;
  std::vector<double> coeffs;

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

template <class Func>
class BarCheb1D {
public:
  BarCheb1D(Func F, const int degree, const double a = -1, const double b = 1)
    : N(degree + 1), a(a), b(b), x(N), w(N), fvals(N) {
    for (int i = 0; i < N; i++) {
      auto c = (2 * i + 1) * PI / (2 * N);
      x[N - i - 1] = map_to_domain(std::cos(c));
      w[N - i - 1] = (1 - 2 * (i % 2)) * std::sin(c);
      fvals[i] = F(x[N - i - 1]);
    }
  }

  double operator()(double pt) const {

    int n = x.size();
    for (int i = 0; i < n; ++i) {
      if (pt == x[i]) { return fvals[i]; }
    }

    double num = 0, den = 0, dif = 0, q = 0;
    for (int i = 0; i < n; ++i) {
      dif = pt - x[i];
      q = w[i] / dif;
      num = num + q * fvals[i];
      den = den + q;
    }

    return num / den;
  }

private:
  int N;
  double a, b;
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
  double a = -.5, b = .5;

  T interpolator(f, degree, a, b);

  std::cout << "Chebyshev interpolation test on random samples:\n";
  std::cout << "Function: f(x) = cos(x), Domain: [" << a << ", " << b << "], Degree: " << degree << "\n\n";

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
    return std::cos(x);
  };
  test<Cheb1D<decltype(f)>>(f);
  std::cout << std::string(80, '-') << "\n";
  std::cout << "\n\n\n";
  test<BarCheb1D<decltype(f)>>(f);
  return 0;
}