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


// 1) Compile-time pow(x, E) via balanced recursion
template <std::size_t E, class T>
constexpr T pow_helper(T x) {
  if constexpr (E == 0)
    return T(1);
  else if constexpr (E == 1)
    return x;
  else if constexpr ((E & 1) == 0) {
    T t = pow_helper<E / 2, T>(x);
    return t * t;
  } else {
    return x * pow_helper<E - 1, T>(x);
  }
}

/* --------------------------------------------------------------------
   constexpr integer power  (k known at compile time)
   ------------------------------------------------------------------*/

[[gnu::always_inline]]
static inline double ipow(double x, std::size_t K) noexcept {

  for (std::size_t i = 1; i < K; i <<= 1) // i = 1, 2, 4, 8, …
    x *= x; // square log2(K) times

  return x; // x now holds x^K
}


// 1) Compile‐time “balanced” exponentiation by squaring
//
// template <std::size_t E, class T>
// constexpr T pow_helper(T x) {
//   if constexpr (E == 0) {
//     return T(1);
//   } else if constexpr (E == 1) {
//     return x;
//   } else if constexpr ((E & 1) == 0) {
//     T t = pow_helper<E / 2, T>(x);
//     return t * t;
//   } else {
//     return x * pow_helper<E - 1, T>(x);
//   }
// }

//
// 2) Build a std::array<T,N> by invoking pow_helper<Is>(x) for Is=0..N-1
//    via index_sequence and a fold‐expression over comma.
//    Batch is the SIMD type (xsimd::batch<T,A>).
//
template <class Batch, std::size_t... Is>
auto x_powers_impl(typename Batch::value_type x, std::index_sequence<Is...>) {
  using T = typename Batch::value_type;
  constexpr std::size_t N = Batch::size;
  alignas(Batch::arch_type::alignment()) std::array<T, N> a{};
  // fold-expression to unroll: a[Is] = pow_helper<Is>(x) for each Is
  ((a[Is] = pow_helper<Is, T>(x)), ...);
  return a;
}

//
// 3) Public entry point: takes a scalar x, returns a batch<T,A> of x^0...x^(N-1)
//
template <class Batch>
Batch x_powers(typename Batch::value_type x) {
  constexpr std::size_t N = Batch::size;
  // build the array
  alignas(Batch::arch_type::alignment()) const auto arr = x_powers_impl<Batch>(x, std::make_index_sequence<N>{});
  // load into SIMD register
  return Batch::load_aligned(arr.data());
}


/**********************************************************************
 * Mix1D –  SIMD Horner blocks + Estrin recombination (vectorised)
 *********************************************************************/
template <class Func>
class Mix1D : public Mon1D<Func> {
public:
  explicit Mix1D(Func F, int n, double a = -1.0, double b = 1.0)
    : Mon1D<Func>(std::move(F), n, a, b) {
    std::reverse(Mon1D<Func>::mono.begin(), Mon1D<Func>::mono.end());
    Mon1D<Func>::mono.resize(padded(n)); // ensure padded size
    // zero the padding lanes
    for (std::size_t i = Mon1D<Func>::N; i < Mon1D<Func>::mono.size(); ++i)
      Mon1D<Func>::mono[i] = 0.0;
  }


  // this works only for high degree polynomials, i.e. N > 64
  double operator()(double pt) const {
    using batch = xsimd::batch<double>;
    constexpr std::size_t W = batch::size;

    const double x = Mon1D<Func>::map_from_domain(pt);

    auto xV = x_powers<batch>(x); // [1 x … x^(W-1)]

    const batch xW_b = xV.get(W - 1) * x;
    const batch xW2_b = xW_b * xW_b;

    const std::size_t blocks = Mon1D<Func>::mono.size() / W;
    const double *coeff = Mon1D<Func>::mono.data();

    batch r0(0.0), r1(0.0); // two independent pipes
    for (std::size_t i = 0; i + 1 < blocks; i += 2) {
      const batch c0 = batch::load_aligned(coeff + i * W);
      const batch c1 = batch::load_aligned(coeff + (i + 1) * W);

      r0 = xsimd::fma(c0, xV, r0); // even block
      r1 = xsimd::fma(c1, xV * xW_b, r1); // odd block

      xV = xV * xW2_b; // advance x^{k+2W}
    }

    if (blocks & 1) {
      // tail block if blocks is odd
      r0 = xsimd::fma(batch::load_aligned(coeff + (blocks - 1) * W), xV, r0);
    }

    return xsimd::reduce_add(r0 + r1); // correct combine!
  }

  // Never really worth it seems
  // Things might change if N is known at compile time
  // double operator()(double pt) const {
  //   /* 1. setup ----------------------------------------------------- */
  //   using batch_type = xsimd::batch<double>;
  //   constexpr std::size_t SIMD = batch_type::size;
  //   const auto numBlocks = Mon1D<Func>::mono.size() / SIMD; // number of SIMD blocks
  //   static_assert(SIMD > 0, "SIMD width must be positive");
  //
  //   const double x = Mon1D<Func>::map_from_domain(pt);
  //
  //   auto xV = x_powers<batch_type>(x);
  //   const auto xSIMD = batch_type(xV.get(SIMD - 1) * x);
  //
  //   batch_type result(0.0); // accumulator for the result
  //
  //   /* 2. SIMD Horner blocks ----------------------------------------- */
  //   for (auto block = 0; block < numBlocks; ++block) {
  //     const auto coeffs = batch_type::load_aligned(Mon1D<Func>::mono.data() + block * SIMD);
  //     result = xsimd::fma(coeffs, xV, result); // accumulate
  //     xV = xV * xSIMD; // update xV for next block
  //   }
  //
  //   return xsimd::reduce_add(result);
  // }

  // Round up to the next multiple of the SIMD width
  // works only for powers of 2
  static constexpr std::size_t padded(const int n) {
    using batch = xsimd::batch<double>;
    constexpr std::size_t simd_width = batch::size;
    return (n + simd_width - 1) & (-simd_width);

  }
};

constexpr std::size_t constexpr_clog2(std::size_t n) {
  return (n < 2 ? 0 : 1 + constexpr_clog2(n / 2));
}

// Recursive Estrin on the range of M coefficients starting at c[I].
// xpows[k] must be x^(2^k), for k=0..constexpr_clog2(N-1).
template <std::size_t I, std::size_t M, typename T>
constexpr T estrin_range(const T *c, const T *xpows) {
  if constexpr (M == 0) {
    return T(0);
  } else if constexpr (M == 1) {
    return c[I];
  } else if constexpr (M == 2) {
    // c[I] + c[I+1] * x
    return std::fma(c[I + 1], xpows[0], c[I]);
  } else {
    // largest power-of-two block <= M-1
    constexpr std::size_t k = constexpr_clog2(M - 1);
    constexpr std::size_t block = std::size_t(1) << k;
    // low  = Estrin(c[I..I+block-1])
    // high = Estrin(c[I+block..I+M-1])
    T low = estrin_range<I, block, T>(c, xpows);
    T high = estrin_range<I + block, M - block, T>(c, xpows);
    // combine: high * x^(block) + low
    return std::fma(high, xpows[k], low);
  }
}

// Public entry: coeffs must be a C-array of length N.
// Builds xpows[0..K] = x^(2^k) by repeated squaring, then calls the
// constexpr-unrolled recursion over [0..N).
template <typename T, std::size_t N>
T est_eval(const T *coeffs, const T x) {
  // we need powers up to x^(2^maxk), where maxk = floor_log2(N-1)
  constexpr std::size_t max_exp = (N > 1 ? N - 1 : 1);
  constexpr std::size_t maxk = constexpr_clog2(max_exp);

  std::array<T, maxk + 1> xpows{};
  xpows[0] = x;
  for (std::size_t k = 1; k <= maxk; ++k) {
    xpows[k] = xpows[k - 1] * xpows[k - 1];
  }

  return estrin_range<0, N, T>(coeffs, xpows.data());
}

// ----------------------------------------------------------------------------
// 2) Build constexpr dispatch table for lengths up to MAX_N
// ----------------------------------------------------------------------------
template <typename T, std::size_t MAX_N, std::size_t... Is>
static constexpr auto make_table(std::index_sequence<Is...>) {
  using Fn = T(*)(const T *, T);
  return std::array<Fn, MAX_N + 1>{&est_eval<T, Is>...};
}

// Runtime dispatcher: picks est_eval<T,n> for actual length n
template <typename T, std::size_t MAX_N>
T est_dispatch(const T *c, std::size_t n, T x) {
  static constexpr auto table =
      make_table<T, MAX_N>(std::make_index_sequence<MAX_N + 1>{});
  if (n > MAX_N)
    throw std::out_of_range("Polynomial length exceeds MAX_N");
  return table[n](c, x);
}

// ----------------------------------------------------------------------------
// 3) Dynamic‐size Estrin wrapper: dimension handled at runtime via est_dispatch
// ----------------------------------------------------------------------------

// Adjust this to the maximum number of coefficients you expect
constexpr std::size_t DEFAULT_MAX_COEFFS = 64;

template <class Func>
class FixedEst : public Mon1D<Func> {
public:
  using Mon1D<Func>::Mon1D;

  explicit FixedEst(Func F, int n, double a = -1.0, double b = 1.0): Mon1D<Func>(F, n, a, b) {
    // Ensure mono is padded to DEFAULT_MAX_COEFFS
    std::reverse(this->mono.begin(), this->mono.end());
  }

  double operator()(double pt) const {
    // 1) map into [-1,1]
    double x = this->map_from_domain(pt);

    // 2) dispatch Estrin on runtime length
    const auto &coeffs = this->mono;
    std::size_t n = coeffs.size();
    return est_dispatch<double, DEFAULT_MAX_COEFFS>(coeffs.data(), n, x);
  }
};