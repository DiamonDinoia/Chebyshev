#include <vector>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <functional>

constexpr double pi = 3.14159265358979323846;

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

// N-dimensional Chebyshev interpolator without mdspan
// Builds coefficients via DCT-I and evaluates via multi-index recursion

template<typename T, size_t N, typename F>
class ChebFunND {
public:
    using point_t = std::array<T, N>;

    ChebFunND(const std::array<int, N>& deg,
              const point_t& a,
              const point_t& b,
              F f)
        : a_(a), b_(b)
    {
        // compute shape and strides
        size_t total = 1;
        for (size_t i = 0; i < N; ++i) {
            shape_[i] = deg[i] + 1;
            total *= shape_[i];
        }
        coeffs_.resize(total);
        strides_[N-1] = 1;
        for (int i = N - 2; i >= 0; --i)
            strides_[i] = strides_[i+1] * shape_[i+1];

        // sample f at Chebyshev nodes
        std::array<size_t, N> idx{};
        sample_recursive<0>(f, idx);
        // apply DCT-I along each axis
        for (size_t axis = 0; axis < N; ++axis)
            dct_axis(axis);
    }

    T operator()(const point_t& x) const {
        // rescale x to [-1,1]
        point_t xt;
        for (size_t i = 0; i < N; ++i)
            xt[i] = (2*x[i] - a_[i] - b_[i]) / (b_[i] - a_[i]);
        // precompute Chebyshev basis
        std::array<std::vector<T>, N> Tvals;
        for (size_t d = 0; d < N; ++d) {
            size_t m = shape_[d];
            Tvals[d].resize(m);
            if (m > 0) Tvals[d][0] = 1;
            if (m > 1) Tvals[d][1] = xt[d];
            for (size_t k = 2; k < m; ++k)
                Tvals[d][k] = 2*xt[d]*Tvals[d][k-1] - Tvals[d][k-2];
        }
        // sum over all multi-indices
        T result = 0;
        std::array<size_t, N> idx{};
        eval_recursive(0, 0, 1, Tvals, result);
        return result;
    }

private:
    point_t a_, b_;
    std::array<size_t, N> shape_{};
    std::array<size_t, N> strides_{};
    std::vector<T> coeffs_;

    // Recursively sample f at Chebyshev nodes
    template<size_t Dim>
    void sample_recursive(F& f, std::array<size_t, N>& idx) {
        for (size_t i = 0; i < shape_[Dim]; ++i) {
            idx[Dim] = i;
            if constexpr (Dim+1 < N)
                sample_recursive<Dim+1>(f, idx);
            else {
                point_t pt;
                for (size_t d = 0; d < N; ++d) {
                    T theta = pi*(idx[d]+0.5)/shape_[d];
                    T xi = std::cos(theta);
                    pt[d] = 0.5*(a_[d]+b_[d]) + 0.5*(b_[d]-a_[d])*xi;
                }
                size_t offset = 0;
                for (size_t d = 0; d < N; ++d)
                    offset += idx[d]*strides_[d];
                coeffs_[offset] = std::apply(f, pt);
            }
        }
    }

    // Apply DCT-I along given axis
    void dct_axis(size_t axis) {
        size_t m = shape_[axis];
        size_t axis_stride = strides_[axis];
        size_t block = axis_stride * m;
        size_t outer = coeffs_.size()/block;
        std::vector<T> buf(m), dct(m);
        for (size_t b = 0; b < outer; ++b) {
            size_t base = b*block;
            for (size_t off = 0; off < axis_stride; ++off) {
                for (size_t j = 0; j < m; ++j)
                    buf[j] = coeffs_[base + off + j*axis_stride];
                for (size_t k = 0; k < m; ++k) {
                    T sum = 0;
                    for (size_t j = 0; j < m; ++j)
                        sum += buf[j]*std::cos(pi*k*(j+0.5)/m);
                    dct[k] = (T(2)/m)*sum;
                }
                dct[0] *= T(0.5);
                for (size_t j = 0; j < m; ++j)
                    coeffs_[base + off + j*axis_stride] = dct[j];
            }
        }
    }

    // Recursively evaluate sum
    void eval_recursive(size_t dim,
                        size_t offset,
                        T prod,
                        const std::array<std::vector<T>, N>& Tvals,
                        T& result) const
    {
        if (dim == N) {
            result += coeffs_[offset] * prod;
            return;
        }
        for (size_t k = 0; k < shape_[dim]; ++k) {
            eval_recursive(dim+1,
                           offset + k*strides_[dim],
                           prod * Tvals[dim][k],
                           Tvals, result);
        }
    }
};

template <size_t D>
void test_dim() {
    using T = double;
    std::array<int, D> deg;
    std::array<T, D> a, b;
    for (size_t i = 0; i < D; ++i) {
        deg[i] = 8;
        a[i] = -1.0;
        b[i] = 1.0;
    }
    auto f = [](auto... xs) {
        T sum = 0;
        ((sum += xs), ...);
        return std::exp(sum);
    };
    ChebFunND<T, D, decltype(f)> cheb(deg, a, b, f);

    // random testing
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<T> dist(-1.0, 1.0);
    double max_rel_err = 0.0;
    const int Ntest = 10000;
    std::array<T, D> pt;
    for (int i = 0; i < Ntest; ++i) {
        double sum = 0;
        auto assign_and_sum = [&](auto& xi) { xi = dist(rng); sum += xi; };
        ([&]<size_t... I>(std::index_sequence<I...>) {
            (assign_and_sum(pt[I]), ...);
        })(std::make_index_sequence<D>{});
        T exact = std::exp(sum);
        T approx = cheb(pt);
        double rel_err = std::abs(approx - exact) / std::abs(exact);
        max_rel_err = std::max(max_rel_err, rel_err);
    }
    std::cout << D << "D max rel err: " << max_rel_err << std::endl;
}

int main() {
    test_dim<1>();
    test_dim<2>();
    test_dim<3>();
    test_dim<4>();
    test_dim<5>();
    test_dim<6>();
    test_dim<7>();
    test_dim<8>();
    return 0;
}
