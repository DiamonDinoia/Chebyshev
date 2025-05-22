#pragma once
#include <algorithm>
#include <vector>
#include <cmath>
#include <functional>


#include <xsimd/xsimd.hpp>


constexpr double PI = 3.14159265358979323846;

template <class Func>
class Cheb1D {
public:
  Cheb1D(Func F, const int n, const double a = -1, const double b = 1)
    : nodes(n), low(b - a), hi(b + a), coeffs(nodes) {

    std::vector<double> fvals(nodes);

    for (int k = 0; k < nodes; ++k) {
      double theta = (2 * k + 1) * PI / (2 * nodes);
      double xk = std::cos(theta);
      double x_mapped = map_to_domain(xk);
      fvals[k] = F(x_mapped);
    }

    for (int m = 0; m < nodes; ++m) {
      double sum = 0.0;
      for (int k = 0; k < nodes; ++k) {
        double theta = (2 * k + 1) * PI / (2 * nodes);
        sum += fvals[k] * std::cos(m * theta);
      }
      coeffs[m] = (2.0 / nodes) * sum;
    }

    coeffs[0] *= 0.5;
    std::reverse(coeffs.begin(), coeffs.end());
  }

  double operator()(const double pt) const {
    const double x = map_from_domain(pt);
    const double x2 = 2 * x;

    double c0 = coeffs[0];
    double c1 = coeffs[1];

    for (int i = 2; i < nodes; ++i) {
      const double tmp = c1;
      c1 = coeffs[i] - c0;
      c0 = c0 * x2 + tmp;
    }

    return c1 + c0 * x;
  }

private:
  const int nodes;
  double low, hi;
  std::vector<double> coeffs;

  template <class T>
  constexpr auto map_to_domain(const T x) const {
    return 0.5 * (low * x + hi);
  }

  constexpr double map_from_domain(double x) const {
    return (2.0 * x - hi) / low;
  }
};

template <class Func>
class BarCheb1D {
public:
  BarCheb1D(Func F, const int n, const double a = -1, const double b = 1)
    : N(n), a(a), b(b), x(padded(N)), w(padded(N)), fvals(padded(N)) {
    for (int i = N - 1; i >= 0; i--) {
      double theta = (2 * i + 1) * PI / (2 * N);
      x[i] = map_to_domain(std::cos(theta));
      w[i] = (1 - 2 * (i % 2)) * std::sin(theta);
      fvals[i] = F(x[i]);
    }
    for (int i = N; i < padded(N); ++i) {
      x[i] = (a + b) * .5;
      w[i] = 0.0;
      fvals[i] = F(x[i]);
    }
  }

  constexpr double operator()(const double pt) const {
    // shorthand for the xsimd type
    using batch = xsimd::batch<double>;
    // simd width since it is architecture/compile flags dependent
    constexpr std::size_t simd_width = batch::size;

    const batch bpt(pt);

    batch bnum(0);
    batch bden(0);

    for (std::size_t i = 0; i < N; i += simd_width) {
      const auto bx = batch::load_aligned(x.data() + i);
      const auto bw = batch::load_aligned(w.data() + i);
      const auto bf = batch::load_aligned(fvals.data() + i);

      if (const auto mask_eq = bx == bpt; xsimd::any(mask_eq)) [[unlikely]] {
        // Return the corresponding fval for the first match
        for (std::size_t j = 0; j < simd_width; ++j) {
          if (mask_eq.get(j)) {
            return bf.get(j);
          }
        }
      }

      const auto bdiff = bpt - bx;
      const auto bq = bw / bdiff;
      bnum = xsimd::fma(bq, bf, bnum);
      bden += bq;
    }

    // Reduce SIMD accumulators to scalars
    const auto num = xsimd::reduce_add(bnum);
    const auto den = xsimd::reduce_add(bden);

    return num / den;
  }

private:
  const int N;
  const double a, b;
  std::vector<double, xsimd::aligned_allocator<double, 64>> x, w, fvals;

  template <class T>
  constexpr auto map_to_domain(const T x) const {
    return 0.5 * ((b - a) * x + (b + a));
  }

  constexpr double map_from_domain(double x) const {
    return (2.0 * x - (b + a)) / (b - a);
  }

  // Round up to the next multiple of the SIMD width
  // works only for powers of 2
  static constexpr std::size_t padded(const int n) {
    using batch = xsimd::batch<double>;
    constexpr std::size_t simd_width = batch::size;
    return (n + simd_width - 1) & (-simd_width);

  }
};