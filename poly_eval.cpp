#include "poly_eval.h"
#include <iostream>
#include <vector>

int main() {
  std::cout << "=== PolyEval Test Suite ===\n";

  // 1) Compile-time sized polynomial p(x) = 1 + 2*x + 3*x^2
  {
    double coeffs[] = {1.0, 2.0, 3.0};
    poly_eval::PolyEval<double, 3> poly(coeffs);
    // Single-point
    double x = 2.0;
    double y = poly(x);
    std::cout << "[CT] poly(2.0) = " << y << " (exp 17)\n";
    // Batch
    double in[] = {0.0, 1.0, 2.0, 3.0, -1.0};
    double out[5];
    poly(in, out, 5);
    std::cout << "[CT] batch out: ";
    for (auto v : out)
      std::cout << v << " ";
    std::cout << "\n";
  }

  // 2) Runtime-sized polynomial q(x) = 5 - x + 0.5*x^3
  {
    std::vector coeffs = {5.0, -1.0, 0.0, 0.5};
    poly_eval::PolyEval poly(coeffs.data(), coeffs.size());
    // Single-point at x = 2: 5 -2 + 0.5*8 = 5 -2 +4 = 7
    double x = 2.0;
    double y = poly(x);
    std::cout << "[RT] poly(2.0) = " << y << " (exp 7)\n";
    // Batch on random points
    std::vector<double, xsimd::aligned_allocator<double, 64>> in(10), out(10);
    for (int i = 0; i < 10; ++i)
      in[i] = -2.0 + i * 0.5;
    poly(in.data(), out.data(), out.size());
    std::cout << "[RT] batch out: ";
    for (auto v : out)
      std::cout << v << " ";
    std::cout << "\n";
  }

  // 3) Comparison against naive Horner
  {
    double coeffs[] = {1, -3, 2}; // p(x)=1 -3x +2x^2
    poly_eval::PolyEval<double, 3> poly(coeffs);
    auto naive = [&](double x) {
      double acc = coeffs[2];
      for (int k = 1; k >= 0; --k)
        acc = acc * x + coeffs[k];
      return acc;
    };
    bool ok = true;
    for (double x : {-2.0, -1.0, 0.0, 1.0, 2.5}) {
      double a = poly(x), b = naive(x);
      if (std::abs(a - b) > 1e-12)
        ok = false;
      std::cout << "cmp(" << x << "): " << a << " vs " << b << "\n";
    }
    std::cout << "Naive comparison: " << (ok ? "PASS" : "FAIL") << "\n";
  }

  std::cout << "=== End of Tests ===\n";
  return 0;
}