#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <cmath>
#include <experimental/mdspan> // or <mdspan> in C++23
#include <functional>
#include <iostream>
#include <random>
#include <type_traits>
#include <utility> // for index_sequence
#include <vector>

#include "fast_eval.hpp" // for function_traits

namespace stdex = std::experimental;
using namespace poly_eval;

// ================================================
// FuncEvalND: N-dimensional, M-output polynomial fit
// with streaming least-squares to avoid large V/Y
// ================================================
template <class Func, std::size_t N_compile_time = 0, std::size_t Iters_compile_time = 1> class FuncEvalND {
public:
  using InputArg0 = typename function_traits<Func>::arg0_type;
  using InputType = std::remove_cvref_t<InputArg0>;
  using OutputType = typename function_traits<Func>::result_type;

  static constexpr std::size_t dim_ = std::tuple_size_v<InputType>;
  static constexpr std::size_t outDim_ = std::tuple_size_v<OutputType>;

  // dynamic-degree constructor only
  template <std::size_t C = N_compile_time, typename = std::enable_if_t<C == 0>>
  FuncEvalND(Func f, int n, InputType const &a, InputType const &b) : func_(f), degree_(n), low_(a), hi_(b) {
    initialize(n, a, b);
  }

  // Evaluate at one point
  OutputType operator()(InputType const &x) const { return evaluate_poly(x); }

private:
  Func func_;
  int degree_;
  InputType low_, hi_;
  std::vector<OutputType> coeffs_; // flat, size = n^dim_

  // mdspan types (column-major so axis-0 fastest)
  using extents_t = stdex::dextents<std::size_t, dim_>;
  using mdspan_t = stdex::mdspan<OutputType, extents_t, stdex::layout_left>;
  using mapping_t = typename mdspan_t::mapping_type;

  mapping_t mapping_;
  mdspan_t coeffs_md_;

  // helper to build extents = {n,n,...}
  template <std::size_t... Is> static extents_t make_extents_impl(int n, std::index_sequence<Is...>) {
    return extents_t{((void)Is, std::size_t(n))...};
  }
  static extents_t make_extents(int n) { return make_extents_impl(n, std::make_index_sequence<dim_>{}); }

  // ------------------------------------------------
  // Streaming initialize: accumulate A = VᵀV, B = VᵀY
  // ------------------------------------------------
  void initialize(int n, InputType const &a, InputType const &b) {
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
    Eigen::MatrixXd C = V.householderQr().solve(`Y);
    // copy into flat coeffs_
    coeffs_.resize(terms);
    for (int i = 0; i < terms; ++i) {
      OutputType tmp{};
      for (size_t d = 0; d < outDim_; ++d)
        tmp[d] = C(i, d);
      coeffs_[i] = tmp;
    }
    // wrap in mdspan
    mapping_ = mapping_t{make_extents(n)};
    coeffs_md_ = mdspan_t{coeffs_.data(), mapping_};
  }

  // ------------------------------------------------
  // Horner evaluation (shared loop)
  // ------------------------------------------------
  template <std::size_t... Is>
  const OutputType &get_coef_impl(const std::array<std::size_t, dim_> &idx, std::index_sequence<Is...>) const {
    return coeffs_md_(idx[Is]...);
  }
  const OutputType &get_coef(const std::array<std::size_t, dim_> &idx) const {
    return get_coef_impl(idx, std::make_index_sequence<dim_>{});
  }

  OutputType evaluate_poly(InputType const &x) const {
    std::array<std::size_t, dim_> idx{};
    return horner<dim_>(x, idx);
  }

  template <std::size_t Rank> OutputType horner(const InputType &x, std::array<std::size_t, dim_> &idx) const {
    // axis = Rank-1, so when Rank==1 we're looping axis=0.
    constexpr size_t axis = Rank - 1;
    OutputType res{};

    for (int k = degree_ - 1; k >= 0; --k) {
      idx[axis] = k;

      // get the “inner” vector: recurse or direct lookup
      OutputType inner;
      if constexpr (Rank > 1) {
        inner = horner<axis>(x, idx);
      } else {
        inner = get_coef(idx);
      }

      // Horner update
      for (size_t d = 0; d < outDim_; ++d) {
        res[d] = res[d] * x[axis] + inner[d];
      }
    }

    return res;
  }
};

int main() {
  // --- choose dims here ---
  constexpr size_t DimIn = 4;
  constexpr size_t DimOut = 4;
  constexpr int N = 8;
  const int Ntest = 1000;

  using VecN = std::array<double, DimIn>;
  using OutM = std::array<double, DimOut>;

  // --- define your function fVec on R^DimIn → R^DimOut ---
  auto fScalar = [](VecN const &x) {
    double s = 0;
    for (double xi : x)
      s += std::sin(xi);
    return s;
  };
  auto fVec = [&](VecN const &x) {
    double r = fScalar(x);
    OutM y{};
    for (size_t i = 0; i < DimOut; ++i)
      y[i] = std::pow(r, i + 1);
    return y;
  };

  // --- domain & degree ---
  VecN a{}, b{};
  a.fill(-1.0);
  b.fill(1.0);

  // --- build approximation ---
  auto t0 = std::chrono::high_resolution_clock::now();
  FuncEvalND<decltype(fVec), 0, 1> approx(fVec, N, a, b);
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "Init: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

  // --- RNG setup ---
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-1, 1);

  // --- benchmark analytical eval & sum ---
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
  double analytic_ms = std::chrono::duration<double, std::milli>(ta1 - ta0).count();
  std::cout << "Analytical eval over " << Ntest << " pts: " << analytic_ms << " ms, sumAnalytic=" << sumAnalytic
            << "\n";

  // --- benchmark polynomial eval & sum ---
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
  double poly_ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();
  std::cout << "Polynomial eval over " << Ntest << " pts: " << poly_ms << " ms, sumPoly=" << sumPoly << "\n";

  // --- compute relative L2 error on same points ---
  double err2 = 0.0, norm2 = 0.0;
  gen.seed(42);
  for (int i = 0; i < Ntest; ++i) {
    VecN x;
    for (auto &xi : x)
      xi = dist(gen);
    auto vE = fVec(x), vP = approx(x);
    for (size_t d = 0; d < DimOut; ++d) {
      double e = vE[d] - vP[d];
      err2 += e * e;
      norm2 += vE[d] * vE[d];
    }
  }
  std::cout << "Relative L2 error: " << std::sqrt(err2 / norm2) << "\n";

  return 0;
}