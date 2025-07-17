#include "fast_eval.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

// Helper function to perform and print error checks in a table format
template <typename TFunc, typename TPoly, typename TInput>
void check_errors(TFunc original_func, TPoly poly_evaluator, TInput domain_a, TInput Input_b,
                  const std::string &description) {
    // Explicitly deduce TOutput from the TPoly's OutputType
    using TOutput = typename TPoly::OutputType;

    std::cout << "\n--- Relative Error Check for " << description << " on [" << domain_a << ", " << Input_b
              << "] ---\n";

    // Set precision for scientific notation for values. Adjusted for better complex number display.
    std::cout << std::scientific << std::setprecision(8);

    // Define column widths for a clean table
    const int INDEX_WIDTH = 5;
    // Increased width for points and values to accommodate complex numbers better
    const int POINT_WIDTH = 30;
    const int VALUE_WIDTH = 17;
    const int ERROR_WIDTH = 22;

    // Print table header
    std::cout << std::setw(INDEX_WIDTH) << std::left << "#";
    std::cout << std::setw(POINT_WIDTH) << std::left << "Random Point (x)";
    std::cout << std::setw(VALUE_WIDTH) << std::left << "Actual Value";
    std::cout << std::setw(VALUE_WIDTH) << std::left << "Poly Value";
    std::cout << std::setw(ERROR_WIDTH) << std::left << "Relative Error (1-|P/A|)";
    std::cout << "\n";
    // Print a separator line
    // Calculate total width of the separator based on column widths
    std::cout << std::string(INDEX_WIDTH + POINT_WIDTH + VALUE_WIDTH * 2 + ERROR_WIDTH, '-') << "\n";

    // Random number generation setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<TInput> dist(domain_a, Input_b);

    for (int i = 0; i < 5; ++i) {
        TInput random_point = dist(gen);
        TOutput actual_val = original_func(random_point);
        TOutput poly_val = poly_evaluator(random_point);

        // Calculate relative error using your specified formula: 1 - |P/A|
        // Note: This formula can yield negative results if |P/A| > 1.
        double rel_err = 1.0 - std::abs(poly_val / actual_val);
        // Print data row
        std::cout << std::setw(INDEX_WIDTH) << std::left << (i + 1);
        std::cout << std::setw(POINT_WIDTH) << std::left << random_point;
        std::cout << std::setw(VALUE_WIDTH) << std::left << actual_val;
        std::cout << std::setw(VALUE_WIDTH) << std::left << poly_val;
        std::cout << std::setw(ERROR_WIDTH) << std::left << rel_err;
        std::cout << "\n";
    }
    std::cout << "\n"; // Add a newline after each table for spacing
}

int main() {
    std::cout << "--- Testing FuncEval with make_func_eval (Error-Driven) API ---\n";

    for (int ns = 2; ns <= 16; ++ns) {
        std::cout << "\n=======================================================\n";
        std::cout << "           Testing for ns = " << ns << "\n";
        std::cout << "=======================================================\n";

        auto target_func = [ns](double x) {
            double beta = 2.30 * ns;
            double c = 4.0 / (static_cast<double>(ns) * ns);
            // Ensure the argument to sqrt is non-negative
            double sqrt_arg = 1.0 - c * x * x;
            return std::exp(beta * (std::sqrt(sqrt_arg) - 1.0));
        };

        double domain_a = -1;
        double domain_b = 1;

        // Set eps_runtime to 10^(-ns + 1)
        double eps_runtime =
            std::max(std::pow(10.0, 1 - static_cast<double>(ns)), std::numeric_limits<double>::epsilon());

        constexpr size_t MAX_N = 32;
        constexpr size_t EVAL_POINTS = 100;
        constexpr size_t ITERS = 1;

        auto poly_evaluator =
            poly_eval::make_func_eval<MAX_N, EVAL_POINTS, ITERS>(target_func, eps_runtime, domain_a, domain_b);

        std::cout << "\nError-Driven Epsilon=" << std::scientific << std::setprecision(2) << eps_runtime
                  << ", MaxN=" << MAX_N << ", EvalPoints=" << EVAL_POINTS << ", Iters=" << ITERS << "):\n";
        std::cout << "  Actual degree found: " << poly_evaluator.coeffs().size() << std::endl;
        // Test at 0.0 (if in domain)
        if (domain_a <= 0.0 && 0.0 <= domain_b) {
            std::cout << "  Poly eval at 0.0: " << poly_evaluator(0.0) << std::endl;
            std::cout << "  Actual at 0.0:    " << target_func(0.0) << std::endl;
        }
        // Test at a point near the domain boundary (e.g., domain_b * 0.9)
        double test_point = domain_b;
        std::cout << "  Poly eval at " << test_point << ": " << poly_evaluator(test_point) << std::endl;
        std::cout << "  Actual at " << test_point << ":    " << target_func(test_point) << std::endl;

        std::string description = "Target Function for ns=" + std::to_string(ns);
        check_errors(target_func, poly_evaluator, domain_a, domain_b, description);
    }

    return 0;
}