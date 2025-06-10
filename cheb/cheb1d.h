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


// Generate n Chebyshev nodes of the first kind on [-1,1]
inline std::vector<double> chebyshev_nodes(int n) {
  std::vector<double> v(n);
  for (int k = 0; k < n; ++k)
    v[k] = std::cos((2.0 * k + 1.0) * PI / (2.0 * n));
  return v;
}

// Affine maps between normalized [-1,1] and [a,b]
inline double map_to_domain(double x, double a, double b) {
  return 0.5 * ((b - a) * x + (b + a));
}

inline double map_from_domain(double x, double a, double b) {
  return (2.0 * x - (b + a)) / (b - a);
}

// Horner evaluation: coeffs in ascending order (c0 + c1*x + ...)
inline double horner(const std::vector<double> &c, double x) {
  if (c.empty())
    return 0.0;
  double y = c.back();
  for (auto it = c.rbegin() + 1; it != c.rend(); ++it)
    y = y * x + *it;
  return y;
}

// Solve symmetric positive-definite system via Gaussian elimination (A square)
inline std::vector<double> solve(std::vector<std::vector<double>> A, std::vector<double> b) {
  int n = (int)A.size();
  assert((int)b.size() == n);
  for (int k = 0; k < n; ++k) {
    // pivot
    int piv = k;
    double maxv = std::abs(A[k][k]);
    for (int i = k + 1; i < n; ++i) {
      double v = std::abs(A[i][k]);
      if (v > maxv) {
        maxv = v;
        piv = i;
      }
    }
    assert(maxv != 0.0 && "Singular normal equations");
    if (piv != k) {
      std::swap(A[piv], A[k]);
      std::swap(b[piv], b[k]);
    }
    // normalize
    double diag = A[k][k];
    for (int j = k; j < n; ++j)
      A[k][j] /= diag;
    b[k] /= diag;
    // eliminate below
    for (int i = k + 1; i < n; ++i) {
      double f = A[i][k];
      for (int j = k; j < n; ++j)
        A[i][j] -= f * A[k][j];
      b[i] -= f * b[k];
    }
  }
  // back-substitute
  std::vector<double> x(n);
  for (int i = n - 1; i >= 0; --i) {
    double s = b[i];
    for (int j = i + 1; j < n; ++j)
      s -= A[i][j] * x[j];
    x[i] = s;
  }
  return x;
}

/* ────────────────────────────────────────────────────────────────
 *  Mon1D - evaluate a function approximated by an n-term Chebyshev
 *  expansion, but converted once to a monomial series and then
 *  evaluated with Horner’s rule.  Public interface is unchanged
 *  except for the new class name.
 * ────────────────────────────────────────────────────────────────*/
template <class Func>
class Mon1D {
public:
  Mon1D(Func F, int n, double a = -1.0, double b = 1.0)
    : N(n), low(b - a), hi(b + a),
      cheb(n), mono(n, 0) // ← mono initialised to zeros
  {
    /* 1. sample F at Chebyshev–Gauss nodes (2k+1)π/2N over [-1,1] */
    std::vector<double> fvals(N);
    for (int k = 0; k < N; ++k) {
      double theta = (2.0 * k + 1.0) * PI / (2.0 * N);
      double xk = std::cos(theta); // on [-1,1]
      fvals[k] = F(map_to_domain(xk));
    }

    /* 2. discrete cosine transform -> Chebyshev coeffs a_m (ascending) */
    for (int m = 0; m < N; ++m) {
      double s = 0.0;
      for (int k = 0; k < N; ++k) {
        double theta = (2.0 * k + 1.0) * PI / (2.0 * N);
        s += fvals[k] * std::cos(m * theta);
      }
      cheb[m] = (2.0 / N) * s;
    }
    cheb[0] *= 0.5; // standard scaling – no reverse!

    /* 3. convert Chebyshev -> monomial once, O(N²) */
    convert_cheb_to_mono();
  }

protected:
  const int N;
  const double low, hi; // low = b-a, hi = b+a
  std::vector<double> cheb; // a_k  for T_k
  std::vector<double, xsimd::aligned_allocator<double>> mono; // b_j  for x^j

  /* ------------------------------------------------------------------
   * convert_cheb_to_mono
   * ------------------------------------------------------------------
   *  Input : cheb[0..N-1]  – Chebyshev coeffs a_k  (ascending k)
   *  Output: mono[0..N-1]  – Power-basis coeffs  b_j  (ascending j)
   *
   *  Algorithm
   *  ---------
   *    Build successive T_k(x) in monomial form via
   *        T_k = 2·x·T_{k-1} - T_{k-2}
   *    and accumulate   a_k · T_k   into the running polynomial.
   *    Work arrays Tkm2 (T_{k-2}) and Tkm1 (T_{k-1}) are reused and
   *    swapped each iteration, so memory is O(N) and flops O(N²).
   *    Basically it uses the three-term recurrence relation for
   *    computing the power series
   *
   *    Finally mono[] is reversed → b_N, … , b_0 so that Horner’s
   *    evaluation processes highest power first.
   * ------------------------------------------------------------------
   */
  void convert_cheb_to_mono() {
    /* ---- k = 0 :  T₀(x) = 1 ------------------------------------ */

    if (N == 1) {
      mono[0] = cheb[0];
      // degree-0 special case
      return;
    }

    /* ---- k = 1 :  T₁(x) = x ------------------------------------ */

    std::vector<double> Tkm1(N, 0);
    std::vector<double> Tkm2(N, 0);
    mono[0] = cheb[0]; // T₀(x) = 1
    mono[1] = cheb[1]; // T₁(x) = x
    Tkm2[0] = 1.0; // monomial 1
    Tkm1[1] = 1.0; // monomial x

    std::vector<double> Tk(N); // will become T_k
    /* ---- k ≥ 2  ------------------------------------------------- */
    for (int k = 2; k < N; ++k) {
      std::fill(Tk.begin(), Tk.begin() + k + 1, 0.0); // clear all N coefficients

      // 1) Tk ← 2·x·T_{k-1} (shift and scale)
      for (int j = 1; j < k + 1; ++j) // T_{k-1} only has terms up to degree k-1
        Tk[j] += 2.0 * Tkm1[j - 1];

      // 2) Tk ← Tk − T_{k−2}
      for (int j = 0; j < k + 1; ++j) // T_{k-2} only has degree k-2
        Tk[j] -= Tkm2[j];

      // 3) Accumulate a_k · T_k
      for (int j = 0; j < k + 1; ++j) {
        mono[j] += cheb[k] * Tk[j];
      }

      std::swap(Tkm2, Tkm1);
      std::swap(Tkm1, Tk);
    }

    /* put mono[] into descending-degree order for Horner */
    std::reverse(mono.begin(), mono.end());
  }

  /* domain maps -------------------------------------------------- */
  template <class T> constexpr T map_to_domain(T x) const { return 0.5 * (low * x + hi); }

  constexpr double map_from_domain(double x) const { return (2.0 * x - hi) / low; }
};


template <class Func>
class Hor1D : public Mon1D<Func> {
public:
  using Mon1D<Func>::Mon1D; // inherit constructor
  /* Horner evaluation on [-1,1] then un-map */
  double operator()(const double pt) const {
    double x = Mon1D<Func>::map_from_domain(pt); // scale to [-1,1]
    double y = 0.0;
    for (int i = 0; i < Mon1D<Func>::N; ++i) {
      y = y * x + Mon1D<Func>::mono[i];
    }
    return y;
  }
};

template <class Func>
class FixedHor : public Mon1D<Func> {
public:
  using Mon1D<Func>::Mon1D;

  double operator()(double pt) const {
    double x = this->map_from_domain(pt);
    return dispatch_eval(this->mono.data(), this->mono.size(), x);
  }

private:
  // maximum supported polynomial length
  static constexpr std::size_t DEFAULT_MAX_COEFFS = 64;

  // Horner’s unrolled recursion: compute c[0] + c[1]*x + … + c[N-1]*x^(N-1)
  // by starting at the highest coefficient and working down.
  // idx = current coefficient index.
  template <std::size_t idx>
  __always_inline static constexpr double horner_step(const double *c, double const x) {
    if constexpr (idx == 0) {
      return c[0];
    } else {
      return std::fma(horner_step<idx - 1>(c, x), x, c[idx]);
    }
  }

  // Evaluate length-N polynomial: calls horner_step<N-1>, or returns 0 if N==0.
  template <std::size_t N>
  __always_inline static double eval_fixed(const double *c, double x) {
    if constexpr (N == 0) {
      return 0.0;
    } else {
      return horner_step<N - 1>(c, x);
    }
  }

  // Build a constexpr table of pointers to eval_fixed<0>, eval_fixed<1>, ….
  template <std::size_t... Is>
  static constexpr auto make_table(std::index_sequence<Is...>) {
    using Fn = double(*)(const double *, double);
    return std::array<Fn, DEFAULT_MAX_COEFFS + 1>{&eval_fixed<Is>...};
  }

  // At first call, instantiates the table; then dispatches in O(1).
  double dispatch_eval(const double *c, std::size_t n, double x) const {
    static constexpr auto table =
        make_table(std::make_index_sequence<DEFAULT_MAX_COEFFS + 1>{});
    if (n > DEFAULT_MAX_COEFFS)
      throw std::out_of_range("Polynomial length exceeds DEFAULT_MAX_COEFFS");
    return table[n](c, x);
  }
};


// Assume Mon1D<Func> is provided elsewhere
template <class Func>
class FixedEst : public Mon1D<Func> {
public:
  explicit FixedEst(Func F,
                    int n,
                    double a = -1.0,
                    double b = 1.0)
    : Mon1D<Func>(F, n, a, b) {
    // reverse to have c[0] first
    std::reverse(this->mono.begin(), this->mono.end());
  }

  double operator()(double pt) const {
    // map into [-1,1]
    const double x = this->map_from_domain(pt);
    // dispatch based on runtime length
    return dispatch_eval(this->mono.data(), this->mono.size(), x);
  }

private:
  // maximum supported degree + 1
  static constexpr std::size_t DEFAULT_MAX_COEFFS = 64;

  //----------------------------------------
  // 1) compile-time helper: floor(log2(n))
  //----------------------------------------
  static constexpr std::size_t clog2(std::size_t n) {
    return (n < 2 ? 0 : 1 + clog2(n / 2));
  }

  //----------------------------------------
  // 2) Estrin-range: unrolled via if constexpr
  //----------------------------------------
  template <std::size_t I, std::size_t M>
  __always_inline static constexpr double estrin_range(const double *c, const double *xpows) {
    if constexpr (M == 0) {
      return 0.0;
    } else if constexpr (M == 1) {
      return c[I];
    } else if constexpr (M == 2) {
      // c[I] + c[I+1]*x  via fmadd: x = xpows[0]
      return std::fma(c[I + 1], xpows[0], c[I]);
    } else {
      constexpr std::size_t k = clog2(M - 1);
      constexpr std::size_t block = std::size_t(1) << k;
      double low = estrin_range<I, block>(c, xpows);
      double high = estrin_range<I + block, M - block>(c, xpows);
      return std::fma(high, xpows[k], low);
    }
  }

  //----------------------------------------
  // 3) Single-call entry: build xpows + recurse
  //----------------------------------------
  template <std::size_t N>
  __always_inline static double eval_fixed(const double *coeffs, double x) {
    constexpr std::size_t max_exp = (N > 1 ? N - 1 : 1);
    constexpr std::size_t maxk = clog2(max_exp);

    std::array<double, maxk + 1> xpows{};
    xpows[0] = x;
    for (std::size_t k = 1; k <= maxk; ++k)
      xpows[k] = xpows[k - 1] * xpows[k - 1];

    return estrin_range<0, N>(coeffs, xpows.data());
  }

  //----------------------------------------
  // 4) Build constexpr dispatch table
  //----------------------------------------
  template <std::size_t... Is>
  static constexpr auto make_table(std::index_sequence<Is...>) {
    using Fn = double(*)(const double *, double);
    return std::array<Fn, DEFAULT_MAX_COEFFS + 1>{&eval_fixed<Is>...};
  }

  //----------------------------------------
  // 5) Runtime dispatcher
  //----------------------------------------
  double dispatch_eval(const double *c, std::size_t n, double x) const {
    static constexpr auto table =
        make_table(std::make_index_sequence<DEFAULT_MAX_COEFFS + 1>{});

    if (n > DEFAULT_MAX_COEFFS)
      throw std::out_of_range("Polynomial length exceeds DEFAULT_MAX_COEFFS");

    return table[n](c, x);
  }
};

template <class Func>
class Est1D : public Mon1D<Func> {
public:
  using Mon1D<Func>::Mon1D; // inherit constructor
  /* Horner evaluation on [-1,1] then un-map */
  double operator()(const double pt) const {
    // 1) map to [-1,1]
    const double x = Mon1D<Func>::map_from_domain(pt);
    const auto &c = Mon1D<Func>::mono; // coeffs, highest-degree first
    const int N = Mon1D<Func>::N; // number of coeffs = degree+1

    // 2) precompute x² and x⁴
    const double x2 = x * x;
    const double x4 = x2 * x2;

    // 3) Estrin‐by‐4 state
    double xpow = 1.0; // holds (x⁴)^j
    double y = 0.0;

    // 4) walk from the constant term backward in steps of 4
    int i;
    for (i = N - 1; i >= 3; i -= 4) {
      // c[i]   = a₀  (constant of this mini-poly)
      // c[i-1] = a₁
      // c[i-2] = a₂
      // c[i-3] = a₃  (highest power in mini-poly)

      // evaluate  a₃·x³ + a₂·x² + a₁·x + a₀ via Horner+FMA:
      double p = std::fma(c[i - 3], x, c[i - 2]); // a₃·x + a₂
      p = std::fma(p, x, c[i - 1]); // (…)*x + a₁
      p = std::fma(p, x, c[i]); // (…)*x + a₀

      // accumulate into y:  y += p * xpow
      y = std::fma(p, xpow, y);

      // bump xpow by x⁴:   xpow *= x4
      xpow = std::fma(xpow, x4, 0.0);
    }

    // 5) handle up to 3 leftover coeffs at the front (highest powers)
    if (i >= 0) {
      // classic Horner on c[0]..c[i]
      double p = c[0];
      for (int j = 1; j <= i; ++j) {
        p = std::fma(p, x, c[j]);
      }
      // accumulate that final piece
      y = std::fma(p, xpow, y);
    }

    return y;
  }
};

template <class Func>
class OptHor : public Mon1D<Func> {
public:
  OptHor(Func F, int n, double a = -1.0, double b = 1.0)
    : Mon1D<Func>(F, n, a, b) {
    // reverse to have c[0] first
    std::reverse(this->mono.begin(), this->mono.end());
  }

  double operator()(double pt) const {
    double x = this->map_from_domain(pt);
    return dispatch_eval(this->mono.data(), this->mono.size(), x);
  }

private:
  /* ------------------------------------------------------------------ */
  static constexpr std::size_t MAX_COEFFS = 64;

  /* fully unrolled scalar Horner (N ≤ 4) ----------------------------- */
  template <std::size_t N, std::size_t I = N>
  __always_inline static constexpr double horner_step(const double *c,
                                                      double x) {
    if constexpr (I == 0)
      return c[0];
    else
      return std::fma(horner_step<N, I - 1>(c, x),
                      x, c[I]);
  }

  template <std::size_t N>
  __always_inline static constexpr double eval_scalar(const double *c,
                                                      double x) {
    if constexpr (N == 0)
      return 0.0;
    else
      return horner_step<N, N - 1>(c, x);
  }

  /* evaluate a *single* 4-term block: c0 + c1 x + c2 x² + c3 x³ ------ */
  __always_inline static double
  eval_block4(const double *c, double x) // c points at c0
  {
    return std::fma(std::fma(std::fma(c[3], x, c[2]), x, c[1]), x, c[0]);
  }

  /* --------------------- corrected 4-way evaluator ------------------ */
  template <std::size_t N>
  __always_inline static double eval_fixed(const double *c, double x) {
    if constexpr (N <= 4) {
      // tiny => scalar Horner
      return eval_scalar<N>(c, x);
    } else {
      /* 1) peel the low-order remainder (N mod 4) ---------------- */
      constexpr std::size_t REM = N & 3;
      double low_part = eval_scalar<REM>(c, x);
      c += REM;

      /* 2) gather the high-order blocks -------------------------- */
      constexpr std::size_t NBLOCK = (N - REM) / 4;
      double acc = eval_block4(c + 4 * (NBLOCK - 1), x); // top block

      /* pre-compute x⁴ once (used to shift each block) ----------- */
      double x2 = x * x;
      double x4 = x2 * x2;

      /* walk remaining blocks top→bottom, shifting by x⁴ each step */
      for (std::size_t b = NBLOCK - 1; b--;) {
        acc = std::fma(acc, x4, // acc * x⁴ + next-blk
                       eval_block4(c + 4 * b, x));
      }

      /* 3) stitch the high- and low-order pieces ---------------- */
      if constexpr (REM == 0) {
        return acc; // no remainder
      } else {
        const auto rem_pow = [x, x2] constexpr {
          if constexpr (REM == 1)
            return x;
          if constexpr (REM == 2)
            return x2;
          if constexpr (REM == 3)
            return x2 * x;
          return 0.0;
        }();
        return std::fma(acc, rem_pow, low_part);
      }
    }
  }

  /* -------------------- zero-overhead dispatch ---------------------- */
  template <std::size_t... Is>
  static constexpr auto make_table(std::index_sequence<Is...>) {
    using Fn = double (*)(const double *, double);
    return std::array<Fn, MAX_COEFFS + 1>{&eval_fixed<Is>...};
  }

  double dispatch_eval(const double *c,
                       std::size_t n,
                       double x) const {
    static constexpr auto tbl =
        make_table(std::make_index_sequence<MAX_COEFFS + 1>{});

    if (n > MAX_COEFFS)
      throw std::out_of_range("Polynomial too long for OptHor");

    return tbl[n](c, x); // fully inlined, branch-free
  }
};