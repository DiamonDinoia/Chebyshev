#include "fast_eval.hpp"

int main() {
    // Create individual evaluators
    const auto sin = poly_eval::make_func_eval([](double x) { return std::sin(x); }, 16, -1.0, 1.0);
    const auto cos = poly_eval::make_func_eval([](double x) { return std::cos(x); }, 16, -1.0, 1.0);

    // Group them via the maker function
    const auto group = poly_eval::make_func_eval(sin, cos);

    double x = 4.0;
    double y = 2.0;
    // Variadic call
    auto results = group(x);
    std::cout << "sin(" << x << ") = " << std::get<0>(results) << ",  cos(" << x << ") = " << std::get<1>(results)
              << '\n';
    std::cout << "Expected sin(" << x << ") = " << std::sin(x) << ",  cos(" << x << ") = " << std::cos(x) << '\n';

    // Tuple call
    auto tuple_args = std::make_tuple(x, y);
    auto tresults = group(tuple_args);
    std::cout << "(tuple) sin = " << std::get<0>(tresults) << ", cos = " << std::get<1>(tresults) << '\n';
    std::cout << "Expected (tuple) sin = " << std::sin(x) << ", cos = " << std::cos(y) << '\n';
    const std::array<double, 2> array_args = {x, y};
    // Array call
    auto aresults = group(array_args);
    std::cout << "(array) sin = " << std::get<0>(aresults) << ", cos = " << std::get<1>(aresults) << '\n';
    std::cout << "Expected (array) sin = " << std::sin(x) << ", cos = " << std::cos(y) << '\n';
    // Variadic arguments call
    auto varesults = group(x, y);
    std::cout << "varesults sin = " << std::get<0>(varesults) << ", cos = " << std::get<1>(varesults) << '\n';
    std::cout << "Expected varesults sin = " << std::sin(x) << ", cos = " << std::cos(y) << '\n';
}
