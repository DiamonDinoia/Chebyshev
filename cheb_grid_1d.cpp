#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// Base template
template <typename T> struct function_traits;

// Function pointer
template <typename Ret, typename... Args> struct function_traits<Ret (*)(Args...)> {
    using return_type = Ret;
    using argument_tuple = std::tuple<Args...>;
    static constexpr std::size_t arity = sizeof...(Args);
    template <std::size_t N> using argument = typename std::tuple_element<N, std::tuple<Args...>>::type;
};

// Member function pointer (const)
template <typename Ret, typename Class, typename... Args> struct function_traits<Ret (Class::*)(Args...) const> {
    using return_type = Ret;
    using argument_tuple = std::tuple<Args...>;
    static constexpr std::size_t arity = sizeof...(Args);
    template <std::size_t N> using argument = typename std::tuple_element<N, std::tuple<Args...>>::type;
};

// Lambda and functor support
template <typename T> struct function_traits : function_traits<decltype(&T::operator())> {};

constexpr double PI = 3.14159265358979323846;

template <typename T> struct ChebGrid1D {
  private:
    using data_t = std::vector<T>;

  public:
    const std::size_t nodes;
    const T a, b;
    const data_t x;
    const data_t w;

    ChebGrid1D(std::size_t n, T a = T(-1), T b = T(1))
        : nodes(n), a(a), b(b), x(init_nodes(n, a, b)), w(init_weights(n)) {}

    constexpr T map_to_domain(T x) const { return T(0.5) * ((b - a) * x + (b + a)); }

    constexpr T map_from_domain(T x) const { return (T(2) * x - (b + a)) / (b - a); }

  private:
    static data_t init_nodes(std::size_t n, T a, T b) {
        data_t result(n);
        for (std::size_t i = 0; i < n; ++i) {
            const auto theta = T(2 * i + 1) * PI / T(2 * n);
            const auto x = std::cos(theta);
            result[i] = T(0.5) * ((b - a) * x + (b + a));
        }
        return result;
    }

    static data_t init_weights(std::size_t n) {
        data_t result(n);
        for (std::size_t i = 0; i < n; ++i) {
            const auto theta = T(2 * i + 1) * PI / T(2 * n);
            result[i] = (T(1) - T(2) * T(i % 2)) * std::sin(theta);
        }
        return result;
    }
};

template <class Func> class Cheb1D {
    using traits = function_traits<Func>;
    using input_t = typename traits::template argument<0>;
    using output_t = typename traits::return_type;

  public:
    Cheb1D(const Func &F, const ChebGrid1D<input_t> &data)
        : nodes(data.nodes), low(data.b - data.a), hi(data.b + data.a), coeffs(nodes) {

        std::vector<output_t> fvals(nodes);
        for (int i = 0; i < nodes; ++i) {
            fvals[i] = F(data.x[i]);
        }

        // Projection using Chebyshev second-kind nodes (DCT-II style)
        for (auto m = 0; m < nodes; ++m) {
            input_t sum = 0.0;
            for (int k = 0; k < nodes; ++k) {
                input_t theta = (2 * k + 1) * PI / (2 * nodes);
                sum += fvals[k] * std::cos(m * theta);
            }
            coeffs[m] = (2.0 / nodes) * sum;
        }

        coeffs[0] *= 0.5; // normalization adjustment
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

    constexpr double map_from_domain(double x) const { return (2.0 * x - hi) / low; }
};

template <class Func> class BarCheb1D {
    using traits = function_traits<Func>;
    using input_t = typename traits::template argument<0>;
    using output_t = typename traits::return_type;

  public:
    BarCheb1D(Func F, const ChebGrid1D<input_t> &data)
        : N(data.nodes), a(data.a), b(data.b), x(data.x), w(data.w), fvals(N) {
        for (int i = 0; i < N; ++i) {
            fvals[i] = F(x[i]);
        }
    }

    double operator()(const double pt) const {
        for (int i = 0; i < N; ++i) {
            if (pt == x[i]) {
                return fvals[i];
            }
        }

        double num = 0, den = 0;
        for (int i = 0; i < N; ++i) {
            double diff = pt - x[i];
            double q = w[i] / diff;
            num += q * fvals[i];
            den += q;
        }

        return num / den;
    }

  private:
    const int N;
    const double a, b;
    const std::vector<double> x, w;
    std::vector<double> fvals;
};

template <typename T, typename V> void test(V &&f) {
    int n = 16;
    double a = -1.5, b = 1.5;

    ChebGrid1D data(n, a, b);
    T interpolator(f, data);

    std::cout << "Chebyshev interpolation test on random samples:\n";
    std::cout << "Function: f(x) = cos(x)+1, Domain: [" << a << ", " << b << "], Nodes: " << n << "\n\n";

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(a, b);

    std::cout << std::setprecision(6) << std::scientific;
    std::cout << "x" << std::setw(20) << "f(x)" << std::setw(20) << "Interp(x)" << std::setw(20) << "Rel. Error\n";
    std::cout << std::string(80, '-') << "\n";

    for (int i = 0; i < 15; ++i) {
        double x = dist(rng);
        double fx = f(x);
        double fx_interp = interpolator(x);
        double err = std::abs(1.0 - fx / fx_interp);
        std::cout << x << "\t" << fx << "\t" << fx_interp << "\t" << err << "\n";
    }
}

int main() {
    auto f = [](double x) { return std::cos(x) + 1; };
    test<Cheb1D<decltype(f)>>(f);
    std::cout << std::string(80, '-') << "\n\n\n";
    test<BarCheb1D<decltype(f)>>(f);
    return 0;
}