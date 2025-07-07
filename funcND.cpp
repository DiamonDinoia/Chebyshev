// ============================================================================
//  FuncEvalND  —  FULL tensor-grid Chebyshev interpolator
// ============================================================================
//  * Fits an N-dimensional, M-output function on the complete tensor product
//    grid of Chebyshev nodes (degree d_i per axis). All cross-terms are kept.
//  * No slice/ prefix logic — we sample once on the whole grid, then compute
//    the coefficients C_{k0..kN-1,o} by the exact multidimensional discrete
//    cosine transform (type-II).  For simplicity (and because degrees are
//    small in practice) we implement the N-D DCT naively via the definition;
//    this keeps the code self-contained and header-only.  You can replace the
//    inner 1-D DCT with FFTW / pocketfft for larger problems.
//  * Evaluation uses pre-computed monomial vectors per axis and a nested loop
//    over all multi-indices k.
//
//  ─────────────  Public interface  ─────────────
//    using ND = FuncEvalND< FuncND /*lambda R^N→R^M*/ >;
//    ND interp( F, low, hi, deg );          // Cheb nodes auto-generated
//    ND interp( F, low, hi, deg, nodes );   // user-supplied Cheb nodes
//    VecM y = interp( x );                  // evaluate at point x∈R^N
//
//  Header-only.  Requires only <vector>, <cmath>, <functional>, <cassert>.
// ============================================================================
#pragma once

#include "utils.h"

#include <cassert>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

namespace poly_eval {

// ---------------------------------------------------------------------------
//  Small helpers
// ---------------------------------------------------------------------------
namespace detail {
// Chebyshev (first-kind) nodes of order n :  x_j = cos(π (j+0.5)/n)
inline std::vector<double> cheb_nodes(std::size_t n) {
  std::vector<double> v(n);
  for (std::size_t j = 0; j < n; ++j)
    v[j] = std::cos(M_PI * (2.0 * j + 1.0) / (2.0 * n));
  return v;
}

// Flatten N-tuple (i0..iN-1) into 1-D index in row-major order.
inline std::size_t flatten(std::vector<std::size_t> const &idx, std::vector<std::size_t> const &dims) {
  std::size_t stride = 1, off = 0;
  for (std::size_t d = 0; d < dims.size(); ++d) {
    off += idx[d] * stride;
    stride *= dims[d];
  }
  return off;
}

// Iterate all multi-indices 0≤k_d<dims[d]  – calls cb(idx).
template <class CB> void for_each_multi(std::vector<std::size_t> const &dims, CB &&cb) {
  std::vector<std::size_t> k(dims.size(), 0);
  while (true) {
    cb(k);
    // increment like odometer
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

// ============================================================================
//  FuncEvalND  TEMPLATE
// ============================================================================

template <class FuncND> class FuncEvalND {
public:
  using VecD = std::vector<double>;                  // length N (inputs)
  using OutVec = std::invoke_result_t<FuncND, VecD>; // length M (outputs)

  // ----------------- ctor (auto nodes) -----------------
  FuncEvalND(FuncND F, VecD low, VecD hi, std::vector<std::size_t> deg)
      : F_(std::move(F)), low_(std::move(low)), hi_(std::move(hi)), deg_(std::move(deg)) {
    init_nodes();
    sample_grid();
    compute_coeffs(); // now computes monomial coefficients
  }

  // ----------------- evaluation -----------------------
  OutVec operator()(VecD const &x) const {
    assert(x.size() == N_);
    // 1. compute monomials x^k per axis
    std::vector<VecD> P(N_);
    for (std::size_t d = 0; d < N_; ++d) {
      std::size_t n = deg_[d] + 1;
      P[d].resize(n);
      P[d][0] = 1.0;
      double xd = (2.0 * x[d] - (low_[d] + hi_[d])) / (hi_[d] - low_[d]); // map to [-1,1]
      for (std::size_t k = 1; k < n; ++k)
        P[d][k] = P[d][k - 1] * xd;
    }

    // 2. tensor contraction: Σ_k coeff(k)*Π_d P[d][k_d]
    OutVec y(M_, 0.0);
    detail::for_each_multi(dims_, [&](std::vector<std::size_t> const &k) {
      double w = 1.0;
      for (std::size_t d = 0; d < N_; ++d)
        w *= P[d][k[d]];
      std::size_t flat = detail::flatten(k, dims_);
      for (std::size_t o = 0; o < M_; ++o)
        y[o] += coeff_[flat][o] * w;
    });
    return y;
  }

private:
  void init_nodes() {
    N_ = low_.size();
    assert(hi_.size() == N_ && deg_.size() == N_);
    nodes_.resize(N_);
    for (std::size_t d = 0; d < N_; ++d)
      nodes_[d] = detail::cheb_nodes(deg_[d] + 1);
    dims_.resize(N_);
    for (std::size_t d = 0; d < N_; ++d)
      dims_[d] = deg_[d] + 1;
    gridSize_ = std::accumulate(dims_.begin(), dims_.end(), std::size_t{1}, std::multiplies<>());
  }

  void sample_grid() {
    sample_.resize(gridSize_);
    VecD tmp = F_(VecD(N_, 0.0));
    M_ = tmp.size();
    detail::for_each_multi(dims_, [&](std::vector<std::size_t> const &idx) {
      VecD x(N_);
      for (std::size_t d = 0; d < N_; ++d) {
        double xi = nodes_[d][idx[d]];
        x[d] = 0.5 * ((hi_[d] - low_[d]) * xi + (hi_[d] + low_[d]));
      }
      sample_[detail::flatten(idx, dims_)] = F_(x);
    });
  }

  void compute_coeffs() {
    // We'll iteratively convert Chebyshev-sampled data into monomial coefficients
    std::vector<OutVec> work = sample_;
    std::vector<OutVec> next(work.size());

    for (std::size_t d = 0; d < N_; ++d) {
      const auto &xs = nodes_[d];
      std::size_t nd = dims_[d];

      // stride of contiguous blocks along axis d
      std::size_t stride = 1;
      for (std::size_t dd = 0; dd < d; ++dd)
        stride *= dims_[dd];
      std::size_t block = stride;
      std::size_t groups = gridSize_ / (block * nd);

      for (std::size_t g = 0; g < groups; ++g) {
        for (std::size_t offset = 0; offset < block; ++offset) {
          // 1-D slice along axis d: gather outputs
          Buffer<OutVec, 0> yvals(nd);
          for (std::size_t j = 0; j < nd; ++j)
            yvals[j] = work[g * block * nd + j * block + offset];

          // Prepare buffer for monomial coeffs of this slice, initialize inner vectors
          Buffer<OutVec, 0> monom(nd, OutVec(M_));

          // For each output component, do scalar Newton->monomial
          for (std::size_t o = 0; o < M_; ++o) {
            // Extract scalar values for this component
            Buffer<double, 0> y_scalar(nd);
            for (std::size_t j = 0; j < nd; ++j)
              y_scalar[j] = yvals[j][o];

            // Divided differences (Newton coeffs)
            auto newton = detail::bjorck_pereyra<0, double, double>(xs, y_scalar);
            // Convert to monomial basis
            auto mono_scalar = detail::newton_to_monomial<0, double, double>(newton, xs);

            // Scatter into monom
            for (std::size_t j = 0; j < nd; ++j)
              monom[j][o] = mono_scalar[j];
          }

          // Scatter monom back into next
          for (std::size_t j = 0; j < nd; ++j)
            next[g * block * nd + j * block + offset] = monom[j];
        }
      }

      work.swap(next);
    }

    coeff_ = std::move(work);
  }

  // data members
  FuncND F_;
  VecD low_, hi_;
  std::vector<std::size_t> deg_;

  std::size_t N_ = 0, M_ = 0;
  std::vector<VecD> nodes_;
  std::vector<std::size_t> dims_;
  std::size_t gridSize_ = 0;

  std::vector<OutVec> sample_;
  std::vector<OutVec> coeff_;
};
} // namespace poly_eval

// ============================================================================
// Optional quick test (compile with -DPOLY_EVAL_ND_TEST)
// ============================================================================
#include <iomanip>
#include <iostream>
int main() {
  using namespace poly_eval;
  using VecD = std::vector<double>;
  auto F = [](VecD const &x) -> VecD { return {x[0] + x[1], x[0] * x[1]}; };
  VecD lo = {-1, -1}, hi = {1, 1};
  std::vector<std::size_t> deg = {4, 4};

  FuncEvalND interp(F, lo, hi, deg);
  std::cout << std::fixed << std::setprecision(4);
  for (double x = -1; x <= 1; x += 0.5)
    for (double y = -1; y <= 1; y += 0.5) {
      auto v = interp({x, y});
      auto e = F({x, y});
      std::cout << "(" << x << "," << y << ") -> {" << v[0] << "," << v[1] << "}  exact {" << e[0] << "," << e[1]
                << "}\n";
    }
  return 0;
}
