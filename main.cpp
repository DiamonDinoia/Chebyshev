#include <vector>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <functional>

constexpr double pi = 3.14159265358979323846;

// Enable scalar multiplication and addition for std::array of size M
template<typename T, size_t M>
std::array<T,M> operator*(const std::array<T,M>& a, T s) {
    std::array<T,M> r;
    for (size_t i = 0; i < M; ++i) r[i] = a[i] * s;
    return r;
}

template<typename T, size_t M>
std::array<T,M>& operator+=(std::array<T,M>& a, const std::array<T,M>& b) {
    for (size_t i = 0; i < M; ++i) a[i] += b[i];
    return a;
}

// 1D Clenshaw for scalar coefficients
template<typename T>
T clenshaw1D(const std::vector<T>& c, T x) {
    int n = static_cast<int>(c.size()) - 1;
    T b_k1 = 0, b_k2 = 0;
    for (int k = n; k >= 1; --k) {
        T b_k = 2 * x * b_k1 - b_k2 + c[k];
        b_k2 = b_k1;
        b_k1 = b_k;
    }
    return x * b_k1 - b_k2 + T(0.5) * c[0];
}

// N-dimensional, M-dimensional-output Chebyshev interpolator
template<typename T, size_t N, size_t M, typename F>
class ChebFunND {
public:
    using point_t = std::array<T, N>;
    using out_t   = std::array<T, M>;

    ChebFunND(const std::array<int, N>& deg,
              const point_t& a,
              const point_t& b,
              F f)
        : a_(a), b_(b)
    {
        // compute shape and strides
        size_t total = 1;
        for (size_t i = 0; i < N; ++i) {
            shape_[i]   = deg[i] + 1;
            total      *= shape_[i];
        }
        coeffs_.resize(total);
        strides_[N-1] = 1;
        for (int i = N-2; i >= 0; --i)
            strides_[i] = strides_[i+1] * shape_[i+1];

        // sample f at Chebyshev nodes
        std::array<size_t, N> idx{};
        sample_recursive<0>(f, idx);

        // apply DCT-I along each axis for each output component
        for (size_t out_i = 0; out_i < M; ++out_i) {
            std::vector<T> comp(total);
            for (size_t i = 0; i < total; ++i) comp[i] = coeffs_[i][out_i];
            for (size_t axis = 0; axis < N; ++axis) {
                size_t m           = shape_[axis];
                size_t axis_stride = strides_[axis];
                size_t block       = axis_stride * m;
                size_t outer       = total / block;
                std::vector<T> buf(m), dct(m);
                for (size_t b = 0; b < outer; ++b) {
                    size_t base = b * block;
                    for (size_t off = 0; off < axis_stride; ++off) {
                        for (size_t j = 0; j < m; ++j)
                            buf[j] = comp[base + off + j*axis_stride];
                        for (size_t k = 0; k < m; ++k) {
                            T sum{};
                            for (size_t j = 0; j < m; ++j)
                                sum += buf[j] * std::cos(pi * k * (j + 0.5) / m);
                            dct[k] = sum * (T(2)/m);
                        }
                        dct[0] *= T(0.5);
                        for (size_t j = 0; j < m; ++j)
                            comp[base + off + j*axis_stride] = dct[j];
                    }
                }
            }
            for (size_t i = 0; i < total; ++i)
                coeffs_[i][out_i] = comp[i];
        }
    }

    // Evaluate at a point, return out_t vector
    out_t operator()(const point_t& x) const {
        point_t xt;
        for (size_t i = 0; i < N; ++i)
            xt[i] = (2 * x[i] - a_[i] - b_[i]) / (b_[i] - a_[i]);

        std::array<std::vector<T>, N> Tvals;
        for (size_t d = 0; d < N; ++d) {
            size_t m = shape_[d];
            Tvals[d].resize(m);
            if (m > 0) Tvals[d][0] = 1;
            if (m > 1) Tvals[d][1] = xt[d];
            for (size_t k = 2; k < m; ++k)
                Tvals[d][k] = 2 * xt[d] * Tvals[d][k-1] - Tvals[d][k-2];
        }

        out_t result{};
        eval_recursive(0, 0, T(1), Tvals, result);
        return result;
    }

private:
    point_t a_, b_;
    std::array<size_t, N> shape_{};
    std::array<size_t, N> strides_{};
    std::vector<out_t> coeffs_;

    // recursive sampling
    template<size_t Dim, typename Func>
    void sample_recursive(Func& f, std::array<size_t, N>& idx) {
        for (size_t i = 0; i < shape_[Dim]; ++i) {
            idx[Dim] = i;
            if constexpr (Dim + 1 < N) sample_recursive<Dim + 1>(f, idx);
            else {
                point_t pt;
                for (size_t d = 0; d < N; ++d) {
                    T theta = pi * (idx[d] + 0.5) / shape_[d];
                    pt[d]   = T(0.5)*(a_[d] + b_[d]) + T(0.5)*(b_[d] - a_[d]) * std::cos(theta);
                }
                size_t off = 0;
                for (size_t d = 0; d < N; ++d) off += idx[d] * strides_[d];
                coeffs_[off] = std::apply(f, pt);
            }
        }
    }

    // recursive evaluation
    void eval_recursive(size_t dim,
                        size_t off,
                        T prod,
                        const std::array<std::vector<T>, N>& Tvals,
                        out_t& acc) const
    {
        if (dim == N) {
            acc += coeffs_[off] * prod;
            return;
        }
        for (size_t k = 0; k < shape_[dim]; ++k)
            eval_recursive(dim+1,
                           off + k*strides_[dim],
                           prod * Tvals[dim][k],
                           Tvals, acc);
    }
};

// Test both N-dimensional inputs and M-dimensional outputs
// for various D and M values
template<size_t D, size_t M>
void test_dim_out() {
    using T = double;
    std::array<int, D> deg{};
    std::array<T, D> a{}, b;
    for (size_t i = 0; i < D; ++i) {
        deg[i] = 8; a[i] = -1; b[i] = 1;
    }
    // f produces M outputs: out[k] = (k+1)*exp(sum(xs))
    auto f = [](auto... xs) {
        T s = 0; ((s += xs), ...);
        std::array<T, M> out;
        for (size_t k = 0; k < M; ++k)
            out[k] = (k+1) * std::exp(s);
        return out;
    };
    ChebFunND<T, D, M, decltype(f)> cheb(deg, a, b, f);

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<T> dist(-1,1);
    std::array<T, D> pt{};
    std::array<double, M> max_rel{};
    max_rel.fill(0);
    const int Ntest = 10000;
    for (int i = 0; i < Ntest; ++i) {
        T sum = 0;
        auto assign = [&](auto& xi){ xi = dist(rng); sum += xi; };
        [&]<std::size_t... I>(std::index_sequence<I...>){ (assign(pt[I]), ...); }(std::make_index_sequence<D>{});
        auto exact = std::array<T, M>();
        for (size_t k = 0; k < M; ++k)
            exact[k] = (k+1) * std::exp(sum);
        auto approx = cheb(pt);
        for (size_t k = 0; k < M; ++k) {
            double err = std::abs(approx[k] - exact[k]) / std::abs(exact[k]);
            max_rel[k] = std::max(max_rel[k], err);
        }
    }
    std::cout << D << "D -> " << M << "D out max rel errors:";
    for (size_t k = 0; k < M; ++k) std::cout << ' ' << max_rel[k];
    std::cout << std::endl;
}

int main() {
    test_dim_out<1,1>();
    test_dim_out<2,1>();
    test_dim_out<3,1>();
    test_dim_out<2,3>();
    test_dim_out<3,2>();
    test_dim_out<4,4>();
    return 0;
}