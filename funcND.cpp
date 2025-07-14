//------------------------------------------------------------------------------
// Tensor-product Björck–Pereyra (any dimension) with nested Horner evaluation
// Benchmark: analytic function vs. polynomial interpolation timing
//------------------------------------------------------------------------------
#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

#include "fast_eval.hpp"
using namespace poly_eval;

/*====================== 1‑D Björck–Pereyra ===============================*/
static void bp_solve(const std::vector<long double> &x, const std::vector<long double> &y,
                     std::vector<long double> &c) {
  int n = static_cast<int>(x.size());
  c = y;
  for (int k = 0; k < n - 1; ++k)
    for (int i = n - 1; i > k; --i)
      c[i] = (c[i] - c[i - 1]) / (x[i] - x[i - k - 1]);
}

/*===================== Nested Horner recursion ===========================*/
// Generic case

template <std::size_t Dim> struct HornerEval {
  static long double eval(const long double *c, const long double *x, const long double *nodes, int n) {
    std::size_t inner = 1;
    for (std::size_t i = 1; i < Dim; ++i)
      inner *= n;
    long double acc = HornerEval<Dim - 1>::eval(c + (n - 1) * inner, x + 1, nodes, n);
    for (int k = n - 2; k >= 0; --k) {
      long double innerVal = HornerEval<Dim - 1>::eval(c + k * inner, x + 1, nodes, n);
      acc = acc * (x[0] - nodes[k]) + innerVal;
    }
    return acc;
  }
};
// Base
template <> struct HornerEval<1> {
  static long double eval(const long double *c, const long double *x, const long double *nodes, int n) {
    long double acc = c[n - 1];
    for (int k = n - 2; k >= 0; --k)
      acc = acc * (x[0] - nodes[k]) + c[k];
    return acc;
  }
};

/*======================== FuncEvalND class ===============================*/

template <class Func> class FuncEvalND {
  using InputArg0 = typename function_traits<Func>::arg0_type;
  using InputType = std::remove_cvref_t<InputArg0>;
  using OutputType = typename function_traits<Func>::result_type;

public:
  static constexpr std::size_t dim_ = std::tuple_size_v<InputType>;
  static constexpr std::size_t outDim_ = std::tuple_size_v<OutputType>;
  FuncEvalND(Func f, int n, const InputType &a, const InputType &b) : func_(f), n_(n), low_(a), hi_(b) { build(); }
  OutputType operator()(const InputType &pt) const { return eval(pt); }

private:
  Func func_;
  int n_;
  InputType low_, hi_;
  std::vector<long double> nodes_;  // n_
  std::vector<long double> coeffs_; // outDim_ * n_^dim_
  static std::size_t ipow(std::size_t b, std::size_t e) {
    std::size_t r = 1;
    while (e--)
      r *= b;
    return r;
  }
  void build() {
    const std::size_t total = ipow(n_, dim_);
    nodes_.resize(n_);
    for (int i = 0; i < n_; ++i)
      nodes_[i] = std::cos(M_PI * (i + 0.5L) / n_);
    // sample
    Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic> Y(total, outDim_);
    for (std::size_t idx = 0; idx < total; ++idx) {
      std::size_t tmp = idx;
      InputType x;
      for (std::size_t d = 0; d < dim_; ++d) {
        int id = tmp % n_;
        tmp /= n_;
        long double t = nodes_[id];
        x[d] = static_cast<double>(0.5L * (low_[d] + hi_[d]) + 0.5L * (hi_[d] - low_[d]) * t);
      }
      auto fx = func_(x);
      for (std::size_t d = 0; d < outDim_; ++d)
        Y(idx, d) = static_cast<long double>(fx[d]);
    }
    coeffs_.resize(total * outDim_);
    std::vector<long double> slice(n_), newton, buf(total);
    for (std::size_t od = 0; od < outDim_; ++od) {
      // copy plane
      for (std::size_t idx = 0; idx < total; ++idx)
        buf[idx] = Y(idx, od);
      // perform BP along each dimension
      for (std::size_t d = 0; d < dim_; ++d) {
        std::size_t stride = ipow(n_, d);
        std::size_t plane = stride * n_;
        std::size_t outer = total / plane;
        for (std::size_t o = 0; o < outer; ++o) {
          std::size_t base_o = o * plane;
          for (std::size_t i = 0; i < stride; ++i) {
            for (int k = 0; k < n_; ++k)
              slice[k] = buf[base_o + i + k * stride];
            bp_solve(nodes_, slice, newton);
            for (int k = 0; k < n_; ++k)
              buf[base_o + i + k * stride] = newton[k];
          }
        }
      }
      for (std::size_t idx = 0; idx < total; ++idx)
        coeffs_[od * total + idx] = buf[idx];
    }
  }
  OutputType eval(const InputType &xin) const {
    long double t[dim_];
    for (std::size_t d = 0; d < dim_; ++d)
      t[d] = (2.0L * xin[d] - low_[d] - hi_[d]) / (hi_[d] - low_[d]);
    // reverse order for Horner layout
    long double tRev[dim_];
    for (std::size_t d = 0; d < dim_; ++d)
      tRev[d] = t[dim_ - 1 - d];
    const std::size_t total = ipow(n_, dim_);
    OutputType y{};
    for (std::size_t od = 0; od < outDim_; ++od) {
      const long double *c = &coeffs_[od * total];
      long double val = HornerEval<dim_>::eval(c, tRev, nodes_.data(), n_);
      y[od] = static_cast<double>(val);
    }
    return y;
  }
};

/*============================== benchmark ================================*/
int main() {
  using Vec3 = std::array<double, 3>;
  auto fScalar = [](const Vec3 &x) {
    return std::exp(std::sin(3.0 * x[0]) * std::cos(2.0 * x[1]) * std::cos(x[2])) + std::cos(x[0] + x[1] - x[2]);
  };
  auto fVec = [&](const Vec3 &x) {
    double r = fScalar(x);
    return Vec3{r, r * r, std::sin(r)};
  };

  Vec3 a{-1.0, -1.0, -1.0}, b{1.0, 1.0, 1.0};
  constexpr int DEG = 16;   // polynomial degree per axis
  const int Ntest = 100000; // benchmarking samples
  // build interpolant
  auto t0 = std::chrono::high_resolution_clock::now();
  FuncEvalND<decltype(fVec)> approx(fVec, DEG, a, b);
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "Init: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

  // random points
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<Vec3> pts(Ntest);
  for (auto &p : pts)
    for (double &c : p)
      c = dist(gen);

  // analytic timing
  auto ta0 = std::chrono::high_resolution_clock::now();
  double norm2 = 0.0;
  for (const auto &p : pts) {
    auto v = fVec(p);
    for (double v_i : v)
      norm2 += v_i * v_i;
  }
  auto ta1 = std::chrono::high_resolution_clock::now();
  double tAnalytic = std::chrono::duration<double, std::milli>(ta1 - ta0).count();

  // polynomial timing + error
  auto tp0 = std::chrono::high_resolution_clock::now();
  double err2 = 0.0;
  for (const auto &p : pts) {
    auto vExact = fVec(p);
    auto vPoly = approx(p);
    for (int d = 0; d < 3; ++d) {
      double e = vExact[d] - vPoly[d];
      err2 += e * e;
    }
  }
  auto tp1 = std::chrono::high_resolution_clock::now();
  double tPoly = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

  std::cout << "Analytic eval:  " << tAnalytic << " ms (" << tAnalytic / Ntest * 1e6 << " µs/pt)\n";
  std::cout << "Poly eval:      " << tPoly << " ms (" << tPoly / Ntest * 1e6 << " µs/pt)\n";
  std::cout << "Rel L2 error:   " << std::sqrt(err2 / norm2) << "\n";
}
