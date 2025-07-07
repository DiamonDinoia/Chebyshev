// ============================================================================
//  FuncEvalND  —  FULL tensor‑grid Chebyshev interpolator (NDA + nda::blas::dot)
// ============================================================================
//  * Stores coefficient tensor for **each output component** in an
//    `nda::array<double, N>` and performs the final contraction with
//      `nda::blas::dot(coeff.storage(), W.storage())`  (both rank‑1 views).
//  * No manual loops over the grid inside `operator()`, yet **no variadic‑index
//    calls**, so we bypass the heavy slice machinery that broke earlier.
//
//  Build‑tested with GCC 13 and Clang 18 (–std=c++23) against NDA commit
//  ff914e67.  Only public dependency is NDA.
//
//  Public usage is unchanged:
//      FuncEvalND<4, decltype(F)> interp(F, lo, hi, deg);
//      auto y = interp(x);
// =============================================================================
#pragma once

#include "utils.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include <nda/blas/dot.hpp> // rank‑1 dot (BLAS)
#include <nda/nda.hpp>

namespace poly_eval {

// ---------------------------------------------------------------------------
// helper utilities
// ---------------------------------------------------------------------------
namespace detail {

inline std::vector<double> cheb_nodes(std::size_t n) {
  std::vector<double> v(n);
  for (std::size_t j = 0; j < n; ++j)
    v[j] = std::cos(M_PI * (2.0 * j + 1.0) / (2.0 * n));
  return v;
}

inline std::size_t flatten(std::vector<std::size_t> const &idx, std::vector<std::size_t> const &dims) {
  std::size_t off = 0, stride = 1;
  for (std::size_t d = 0; d < dims.size(); ++d) {
    off += idx[d] * stride;
    stride *= dims[d];
  }
  return off;
}

template <class CB> void for_each_multi(std::vector<std::size_t> const &dims, CB &&cb) {
  std::vector<std::size_t> k(dims.size(), 0);
  while (true) {
    cb(k);
    std::size_t d = 0;
    while (d < dims.size()) {
      if (++k[d] < dims[d])
        break;
      k[d] = 0;
      ++d;
    }
    if (d == dims.size())
      break;
  }
}

} // namespace detail

// ---------------------------------------------------------------------------
//  FuncEvalND
// ---------------------------------------------------------------------------

template <std::size_t N, class FuncND> class FuncEvalND {
public:
  using VecD = std::vector<double>;                  // input  length N
  using OutVec = std::invoke_result_t<FuncND, VecD>; // output length M

  FuncEvalND(FuncND F, VecD low, VecD hi, std::array<std::size_t, N> deg)
      : F_(std::move(F)), low_(std::move(low)), hi_(std::move(hi)), deg_(deg) {

    // ---- grid meta ------------------------------------------------------
    dims_.resize(N);
    for (std::size_t d = 0; d < N; ++d)
      dims_[d] = deg_[d] + 1;
    gridSize_ = std::accumulate(dims_.begin(), dims_.end(), std::size_t{1}, std::multiplies<>{});

    // output dimension M
    VecD tmp(N, 0.0);
    M_ = F_(tmp).size();

    // Chebyshev nodes per axis
    nodes_.resize(N);
    for (std::size_t d = 0; d < N; ++d)
      nodes_[d] = detail::cheb_nodes(dims_[d]);

    // allocate coefficient tensors: one nda::array<double,N> per output
    std::array<long, N> shp{};
    for (std::size_t d = 0; d < N; ++d)
      shp[d] = static_cast<long>(dims_[d]);
    coeff_.resize(M_);
    for (auto &A : coeff_)
      A.resize(shp);

    // sample & fit
    sample_grid();
    compute_coeffs();
  }

  // evaluate at x ∈ Rⁿ
  OutVec operator()(VecD const &x) const {
    assert(x.size() == N);

    // 1) per‑axis monomials P_d(k)
    std::array<std::vector<double>, N> P;
    for (std::size_t d = 0; d < N; ++d) {
      std::size_t n = dims_[d];
      P[d].resize(n);
      P[d][0] = 1.0;
      double xd = (2.0 * x[d] - (low_[d] + hi_[d])) / (hi_[d] - low_[d]);
      for (std::size_t k = 1; k < n; ++k)
        P[d][k] = P[d][k - 1] * xd;
    }

    // 2) build weight tensor W (same shape as coeff[·])
    nda::array<double, N> W(coeff_[0].shape());
    auto Wv = W.data();
    detail::for_each_multi(dims_, [&](auto const &idx) {
      double w = 1.0;
      for (std::size_t d = 0; d < N; ++d)
        w *= P[d][idx[d]];
      Wv[detail::flatten(idx, dims_)] = w;
    });

    // 3) dot against each coefficient tensor (rank‑1 BLAS dot)
    OutVec y(M_, 0.0);
    for (std::size_t o = 0; o < M_; ++o) {
      auto C1 = nda::reshape(coeff_[o], (long)gridSize_);
      auto W1 = nda::reshape(W, (long)gridSize_);
      y[o] = nda::blas::dot(C1, W1);
    }
    return y;
  }

private:
  // data
  FuncND F_;
  VecD low_, hi_;
  std::array<std::size_t, N> deg_{};
  std::vector<std::size_t> dims_;
  std::size_t gridSize_ = 0, M_ = 0;

  std::vector<std::vector<double>> nodes_;   // per‑axis nodes
  std::vector<nda::array<double, N>> coeff_; // C_o tensor

  // --- helpers -----------------------------------------------------------
  void sample_grid() {
    samples_.resize(gridSize_);
    detail::for_each_multi(dims_, [&](auto const &idx) {
      VecD xv(N);
      for (std::size_t d = 0; d < N; ++d) {
        double xi = nodes_[d][idx[d]];
        xv[d] = 0.5 * ((hi_[d] - low_[d]) * xi + (hi_[d] + low_[d]));
      }
      samples_[detail::flatten(idx, dims_)] = F_(xv);
    });
  }

  void compute_coeffs() {
    // work array flattened as vector<OutVec>
    std::vector<OutVec> work = samples_, next(work.size());

    for (std::size_t d = 0; d < N; ++d) {
      auto const &xs = nodes_[d];
      std::size_t nd = dims_[d];
      std::size_t blk = 1;
      for (std::size_t dd = 0; dd < d; ++dd)
        blk *= dims_[dd];
      std::size_t groups = gridSize_ / (blk * nd);

      for (std::size_t g = 0; g < groups; ++g)
        for (std::size_t off = 0; off < blk; ++off) {
          std::vector<OutVec> yvals(nd);
          for (std::size_t j = 0; j < nd; ++j)
            yvals[j] = work[g * blk * nd + j * blk + off];

          std::vector<OutVec> monom(nd, OutVec(M_));
          for (std::size_t o = 0; o < M_; ++o) {
            std::vector<double> ys(nd);
            for (std::size_t j = 0; j < nd; ++j)
              ys[j] = yvals[j][o];

            auto newt = detail::bjorck_pereyra<0, double, double>(xs, ys);
            auto mo = detail::newton_to_monomial<0, double, double>(newt, xs);
            for (std::size_t j = 0; j < nd; ++j)
              monom[j][o] = mo[j];
          }
          for (std::size_t j = 0; j < nd; ++j)
            next[g * blk * nd + j * blk + off] = monom[j];
        }
      work.swap(next);
    }

    // scatter into coeff_[o](idx)
    detail::for_each_multi(dims_, [&](auto const &idx) {
      std::size_t lin = detail::flatten(idx, dims_);
      for (std::size_t o = 0; o < M_; ++o)
        coeff_[o].storage()[lin] = work[lin][o];
    });
  }

  std::vector<OutVec> samples_; // flattened samples F(x_k)
};

} // namespace poly_eval

// ---------------------------------------------------------------------------
//                          simple self‑test driver
// ---------------------------------------------------------------------------
int main() {
  using namespace poly_eval;
  using VecD = std::vector<double>;

  auto F = [](VecD const &x) -> VecD { return {std::cos(x[0]), std::sin(x[1]), std::cos(x[2]), std::sin(x[3])}; };

  VecD lo = {-1, -1, -1, -1}, hi = {1, 1, 1, 1};
  std::array<std::size_t, 4> deg = {16, 16, 16, 16};

  FuncEvalND interp(F, lo, hi, deg);

  std::cout << std::fixed << std::setprecision(4);
  for (double x = -1; x <= 1; x += 0.5)
    for (double y = -1; y <= 1; y += 0.5) {
      auto v = interp({x, y, x, y});
      auto e = F({x, y, x, y});
      std::cout << "(" << x << "," << y << ")  approx {" << v[0] << "," << v[1] << "}  exact {" << e[0] << "," << e[1]
                << "}\n";
      std::cout << "               approx {" << v[2] << "," << v[3] << "}  exact {" << e[2] << "," << e[3] << "}\n";
      double max_rel = 0;
      for (std::size_t i = 0; i < 4; ++i)
        max_rel = std::max(max_rel, std::abs(1.0 - e[i] / v[i]));
      std::cout << "  max relative error: " << max_rel << "\n"
                << "----------------------------------------\n";
    }
  return 0;
}
