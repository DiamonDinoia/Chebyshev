#include <vector>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <functional>

constexpr double pi = 3.14159265358979323846;

template <typename T, size_t M>
std::array<T, M> operator*(const std::array<T, M> &a, T s) {
  std::array<T, M> r;
  for (size_t i = 0; i < M; ++i)
    r[i] = a[i] * s;
  return r;
}

template <typename T, size_t M>
std::array<T, M> &operator+=(std::array<T, M> &a, const std::array<T, M> &b) {
  for (size_t i = 0; i < M; ++i)
    a[i] += b[i];
  return a;
}

template <typename T, size_t N, size_t M, typename F, int... Degrees>
class ChebFunND {
  static_assert(sizeof...(Degrees) == N, "Degree count must match dimensionality");

public:
  static constexpr std::array<size_t, N> shape = { static_cast<size_t>(Degrees + 1)... };

  static constexpr std::array<size_t, N> compute_strides() {
    std::array<size_t, N> s{};
    s[N - 1] = 1;
    for (int i = N - 2; i >= 0; --i)
      s[i] = s[i + 1] * shape[i + 1];
    return s;
  }

  static constexpr std::array<size_t, N> strides = compute_strides();

  static constexpr size_t total_size = [] {
    size_t t = 1;
    for (auto v : shape) t *= v;
    return t;
  }();

  using point_t = std::array<T, N>;
  using out_t = std::array<T, M>;

  ChebFunND(const point_t &a, const point_t &b, F f)
    : a_(a), b_(b), coeffs_(total_size) {
    std::array<size_t, N> idx{};
    sample<0>(f, idx);

    for (size_t out_i = 0; out_i < M; ++out_i) {
      std::vector<T> comp(total_size);
      for (size_t i = 0; i < total_size; ++i)
        comp[i] = coeffs_[i][out_i];

      for (size_t axis = 0; axis < N; ++axis) {
        size_t m = shape[axis];
        size_t axis_stride = strides[axis];
        size_t block = axis_stride * m;
        size_t outer = total_size / block;
        std::vector<T> buf(m), dct(m);

        for (size_t b = 0; b < outer; ++b) {
          size_t base = b * block;
          for (size_t off = 0; off < axis_stride; ++off) {
            for (size_t j = 0; j < m; ++j)
              buf[j] = comp[base + off + j * axis_stride];
            for (size_t k = 0; k < m; ++k) {
              T sum{};
              for (size_t j = 0; j < m; ++j)
                sum += buf[j] * std::cos(pi * k * (j + 0.5) / m);
              dct[k] = sum * (T(2) / m);
            }
            dct[0] *= T(0.5);
            for (size_t j = 0; j < m; ++j)
              comp[base + off + j * axis_stride] = dct[j];
          }
        }
      }

      for (size_t i = 0; i < total_size; ++i)
        coeffs_[i][out_i] = comp[i];
    }
  }

  out_t operator()(const point_t &x) const {
    point_t xt;
    for (size_t i = 0; i < N; ++i)
      xt[i] = (2 * x[i] - a_[i] - b_[i]) / (b_[i] - a_[i]);

    std::array<std::vector<T>, N> Tvals;
    for (size_t d = 0; d < N; ++d) {
      size_t m = shape[d];
      Tvals[d].resize(m);
      if (m > 0) Tvals[d][0] = 1;
      if (m > 1) Tvals[d][1] = xt[d];
      for (size_t k = 2; k < m; ++k)
        Tvals[d][k] = 2 * xt[d] * Tvals[d][k - 1] - Tvals[d][k - 2];
    }

    out_t result{};
    eval(0, 0, T(1), Tvals, result);
    return result;
  }

  template <typename... Args>
  out_t operator()(Args... args) const {
    return (*this)(std::array<T, N>{static_cast<T>(args)...});
  }

private:
  point_t a_, b_;
  std::vector<out_t> coeffs_;

  template <size_t Dim, typename Func>
  void sample(Func &f, std::array<size_t, N> &idx) {
    for (size_t i = 0; i < shape[Dim]; ++i) {
      idx[Dim] = i;
      if constexpr (Dim + 1 < N)
        sample<Dim + 1>(f, idx);
      else {
        point_t pt;
        for (size_t d = 0; d < N; ++d) {
          T theta = pi * (idx[d] + 0.5) / shape[d];
          pt[d] = T(0.5) * (a_[d] + b_[d]) + T(0.5) * (b_[d] - a_[d]) * std::cos(theta);
        }
        size_t off = 0;
        for (size_t d = 0; d < N; ++d)
          off += idx[d] * strides[d];
        coeffs_[off] = std::apply(f, pt);
      }
    }
  }

  void eval(size_t dim, size_t off, T prod,
            const std::array<std::vector<T>, N> &Tvals,
            out_t &acc) const {
    if (dim == N) {
      acc += coeffs_[off] * prod;
      return;
    }
    for (size_t k = 0; k < shape[dim]; ++k)
      eval(dim + 1, off + k * strides[dim], prod * Tvals[dim][k], Tvals, acc);
  }
};

// === Test Harness ===
template <size_t D, size_t M, typename ChebType, typename F>
void test_interp(const std::array<double, D>& a,
                 const std::array<double, D>& b,
                 F f) {
  ChebType cheb(a, b, f);
  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> dist(-1, 1);

  std::array<double, D> pt{};
  std::array<double, M> max_rel{};
  max_rel.fill(0);
  const int Ntest = 10000;

  for (int i = 0; i < Ntest; ++i) {
    double sum = 0;
    auto assign = [&](auto& xi) {
      xi = dist(rng);
      sum += xi;
    };
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      (assign(pt[I]), ...);
    }(std::make_index_sequence<D>{});

    std::array<double, M> exact{};
    for (size_t k = 0; k < M; ++k)
      exact[k] = (k + 1) * std::exp(sum);

    auto approx = cheb(pt);
    for (size_t k = 0; k < M; ++k)
      max_rel[k] = std::max(max_rel[k], std::abs(1.0 - approx[k] / exact[k]));
  }

  std::cout << D << "D -> " << M << "D out max rel errors:";
  for (double e : max_rel)
    std::cout << ' ' << e;
  std::cout << '\n';
}

template <size_t D, size_t M>
void test_dim_out() {
  using T = double;
  std::array<T, D> a{}, b;
  for (size_t i = 0; i < D; ++i) {
    a[i] = -1;
    b[i] = 1;
  }
  auto f = [](auto... xs) {
    T s = 0;
    ((s += xs), ...);
    std::array<T, M> out{};
    for (size_t k = 0; k < M; ++k)
      out[k] = (k + 1) * std::exp(s);
    return out;
  };
  if constexpr (D == 1)
    test_interp<D, M, ChebFunND<T, D, M, decltype(f), 8>>(a, b, f);
  else if constexpr (D == 2)
    test_interp<D, M, ChebFunND<T, D, M, decltype(f), 8, 8>>(a, b, f);
  else if constexpr (D == 3)
    test_interp<D, M, ChebFunND<T, D, M, decltype(f), 8, 8, 8>>(a, b, f);
  else if constexpr (D == 4)
    test_interp<D, M, ChebFunND<T, D, M, decltype(f), 8, 8, 8, 8>>(a, b, f);
}

int main() {
  test_dim_out<1, 1>();
  test_dim_out<2, 1>();
  test_dim_out<3, 1>();
  test_dim_out<2, 3>();
  test_dim_out<3, 2>();
  test_dim_out<4, 4>();
  return 0;
}