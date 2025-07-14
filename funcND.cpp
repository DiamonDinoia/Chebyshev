#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

// Bring in function_traits and Buffer from poly_eval
#include "fast_eval.hpp"

#include <chrono>
using namespace poly_eval;

// FuncEvalND: fits a vector-valued multivariate polynomial via least squares
// Template parameters:
//  Func               - callable InputType -> OutputType
//  N_compile_time     - compile-time degree (0 => dynamic)
//  Iters_compile_time - number of refinement passes (optional)

template <class Func, std::size_t N_compile_time = 0, std::size_t Iters_compile_time = 1> class FuncEvalND {
public:
  using InputArg0 = typename function_traits<Func>::arg0_type;
  using InputType = std::remove_cvref_t<InputArg0>;
  using OutputType = typename function_traits<Func>::result_type;

  static constexpr std::size_t kDegreeCompileTime = N_compile_time;
  static constexpr std::size_t kItersCompileTime = Iters_compile_time;

  // dynamic-degree constructor
  template <std::size_t Cur = N_compile_time, typename = std::enable_if_t<Cur == 0>>
  FuncEvalND(Func f, int n, const InputType &a, const InputType &b) : func_(f), degree_(n), low_(a), hi_(b) {
    initialize(n, a, b);
  }

  // evaluate at a single point
  OutputType operator()(const InputType &pt) const { return evaluate_poly(pt); }

private:
  Func func_;
  int degree_;
  static constexpr std::size_t dim_ = std::tuple_size_v<InputType>;
  static constexpr std::size_t outDim_ = std::tuple_size_v<OutputType>;
  InputType low_, hi_;
  Buffer<OutputType, N_compile_time> coeffs_;

  // build sample matrix and solve for vector coefficients
  void initialize(int n, const InputType &a, const InputType &b) {
    const int samples = 2 * n;
    int total = 1;
    for (std::size_t i = 0; i < dim_; ++i)
      total *= samples;
    int terms = 1;
    for (std::size_t i = 0; i < dim_; ++i)
      terms *= n;

    Eigen::MatrixXd V(total, terms);
    Eigen::MatrixXd Y(total, outDim_);

    // Chebyshev nodes
    std::vector<double> nodes(samples);
    for (int i = 0; i < samples; ++i)
      nodes[i] = std::cos(M_PI * (i + 0.5) / samples);

    // fill V and Y
    for (int idx = 0; idx < total; ++idx) {
      int tmp = idx;
      InputType x;
      for (std::size_t d = 0; d < dim_; ++d) {
        int id = tmp % samples;
        tmp /= samples;
        double t = nodes[id];
        x[d] = 0.5 * (a[d] + b[d]) + 0.5 * (b[d] - a[d]) * t;
      }
      auto fx = func_(x); // OutputType vector
      for (std::size_t d = 0; d < outDim_; ++d)
        Y(idx, d) = fx[d];

      for (int mon = 0; mon < terms; ++mon) {
        int code = mon;
        double pval = 1.0;
        for (std::size_t d = 0; d < dim_; ++d) {
          int p = code % n;
          code /= n;
          pval *= std::pow(x[d], p);
        }
        V(idx, mon) = pval;
      }
    }

    // solve V * C = Y in least squares sense
    const auto A = V.transpose() * V; // n×n
    const auto B = V.transpose() * Y; // n×p
    // 2) factor & solve
    Eigen::LDLT<Eigen::MatrixXd> ldlt(A); //
    Eigen::MatrixXd C = ldlt.solve(B);    // n×p result
    coeffs_.clear();
    coeffs_.resize(terms);

    // store term-wise coefficient vectors
    for (int i = 0; i < terms; ++i) {
      OutputType cvec{};
      for (std::size_t d = 0; d < outDim_; ++d)
        cvec[d] = C(i, d);
      coeffs_[i] = cvec;
    }
  }

  // evaluate polynomial sum_i coeffs_[i] * monomial(x,i)
  OutputType evaluate_poly(const InputType &x) const {
    OutputType y{};
    int terms = static_cast<int>(coeffs_.size());
    for (int i = 0; i < terms; ++i) {
      double m = monomial(x, i);
      // accumulate vector result
      for (std::size_t d = 0; d < outDim_; ++d)
        y[d] += coeffs_[i][d] * m;
    }
    return y;
  }

  // compute monomial index->value
  double monomial(const InputType &x, int idx) const {
    double prod = 1.0;
    int code = idx;
    for (std::size_t d = 0; d < dim_; ++d) {
      int p = code % degree_;
      code /= degree_;
      prod *= std::pow(x[d], p);
    }
    return prod;
  }
};

int main() {
  using Vec3 = std::array<double, 3>;

  // scalar function f: R^3->R
  auto fScalar = [](const Vec3 &x) {
    return std::exp(std::sin(3.0 * x[0]) * std::cos(2.0 * x[1]) * std::cos(x[2])) + std::cos(x[0] + x[1] - x[2]);
  };
  // vector-valued function fVec: R^3->R^3
  auto fVec = [&](const Vec3 &x) {
    double r = fScalar(x);
    return Vec3{r, r * r, std::sin(r)};
  };

  Vec3 a{-1.0, -1.0, -1.0}, b{1.0, 1.0, 1.0};
  constexpr int N = 16;
  const int Ntest = 1000;

  // Time the initialization (fitting)
  auto t_init_start = std::chrono::high_resolution_clock::now();
  FuncEvalND<decltype(fVec), 0, 1> approx(fVec, N, a, b);
  auto t_init_end = std::chrono::high_resolution_clock::now();
  auto init_ms = std::chrono::duration<double, std::milli>(t_init_end - t_init_start).count();
  std::cout << "Initialization time: " << init_ms << " ms" << std::endl;

  // Prepare random data
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  double err2 = 0.0, norm2 = 0.0;

  // Time the evaluation loop
  auto t_eval_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < Ntest; ++i) {
    Vec3 x;
    for (int d = 0; d < 3; ++d)
      x[d] = dist(gen);
    auto vExact = fVec(x);
    auto vPoly = approx(x);
    for (int d = 0; d < 3; ++d) {
      double e = vExact[d] - vPoly[d];
      err2 += e * e;
      norm2 += vExact[d] * vExact[d];
    }
  }
  auto t_eval_end = std::chrono::high_resolution_clock::now();
  auto eval_ms = std::chrono::duration<double, std::milli>(t_eval_end - t_eval_start).count();
  std::cout << "Evaluation time: " << eval_ms << " ms" << std::endl;
  // Report results
  std::cout << "Total time: " << init_ms + eval_ms << " ms";
  std::cout << "Relative L2 error: " << std::sqrt(err2 / norm2) << std::endl;
  return 0;
}
