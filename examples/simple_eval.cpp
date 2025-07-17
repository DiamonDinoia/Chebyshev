#include "fast_eval.hpp" // Assuming your poly_eval.h is renamed to fast_eval.h

#include <cmath> // For std::cos, std::sin, std::abs
#include <complex>
#include <iomanip> // For std::scientific, std::setprecision, std::setw, std::left, std::right
#include <iostream>
#include <limits> // For std::numeric_limits
#include <random> // For std::random_device, std::mt19937, std::uniform_real_distribution
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
    const int VALUE_WIDTH = 35; // Significantly increased to fit complex<double> output
    const int ERROR_WIDTH = 22;

    // Print table header
    std::cout << std::setw(INDEX_WIDTH) << std::left << "#";
    std::cout << std::setw(POINT_WIDTH) << std::left << "Random Point (x)";
    std::cout << std::setw(VALUE_WIDTH) << std::left << "Actual Value";
    std::cout << std::setw(VALUE_WIDTH) << std::left << "Poly Value";
    std::cout << std::setw(ERROR_WIDTH) << std::left << "Relative Error (|1-P/A|)";
    std::cout << "\n";
    // Print a separator line
    // Calculate total width of the separator based on column widths
    std::cout << std::string(INDEX_WIDTH + POINT_WIDTH + VALUE_WIDTH * 2 + ERROR_WIDTH, '-') << "\n";

    // Random number generation setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<TInput> dist(domain_a, Input_b);

    for (int i = 0; i < 10; ++i) {
        TInput random_point = dist(gen);
        TOutput actual_val = original_func(random_point);
        TOutput poly_val = poly_evaluator(random_point);

        // Calculate relative error using your specified formula: 1 - |P/A|
        // Note: This formula can yield negative results if |P/A| > 1.
        double rel_err = std::abs(1.0 - poly_val / actual_val);
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
    // Define some example functions
    auto my_func_double = [](double x) { return std::cos(2 * x); };
    auto my_func_float = [](float x) { return std::sin(x) + std::cos(x); };
    auto my_func_complex = [](double x) { return std::complex<double>(x * x, std::sin(x)); };

    std::cout << "--- Testing FuncEval with make_func_eval API ---\n";

    // -----------------------------------------------------
    // 1. Runtime Degree (n), Default Iterations (1)
    // -----------------------------------------------------
    double domain_a1 = -.5;
    double domain_b1 = .5;
    auto poly_runtime_d_default_iters = poly_eval::make_func_eval(my_func_double, 16, domain_a1, domain_b1);
    std::cout << "\nRuntime Degree (double, n=16, iters=1):\n";
    std::cout << "Poly eval at 0.0: " << poly_runtime_d_default_iters(0.0) << std::endl;
    std::cout << "Actual at 0.0:    " << my_func_double(0.0) << std::endl;
    std::cout << "Coefficients count: " << poly_runtime_d_default_iters.coeffs().size() << std::endl;

    // Call the helper for error checking
    check_errors(my_func_double, poly_runtime_d_default_iters, domain_a1, domain_b1, "my_func_double (n=16, iters=1)");

    // -----------------------------------------------------
    // 2. Runtime Degree (n), Custom Compile-Time Iterations
    // -----------------------------------------------------
    float domain_a2 = -static_cast<float>(M_PI);
    float domain_b2 = static_cast<float>(M_PI);
    auto poly_runtime_f_3iters = poly_eval::make_func_eval<1>(my_func_float, 8, domain_a2, domain_b2);
    std::cout << "\nRuntime Degree (float, n=8, iters=1):\n";
    std::cout << "Poly eval at 0.0f: " << poly_runtime_f_3iters(0.0f) << std::endl;
    std::cout << "Actual at 0.0f:    " << my_func_float(0.0f) << std::endl;
    std::cout << "Coefficients count: " << poly_runtime_f_3iters.coeffs().size() << std::endl;

    // Call the helper for error checking
    check_errors(my_func_float, poly_runtime_f_3iters, domain_a2, domain_b2, "my_func_float (n=8, iters=1)");

    // -----------------------------------------------------
    // 3. Compile-Time Degree (N), Default Iterations (1)
    // -----------------------------------------------------
    double domain_a3 = -5.0;
    double domain_b3 = 5.0;
    auto poly_compiletime_d_default_iters = poly_eval::make_func_eval<6>(my_func_double, domain_a3, domain_b3);
    std::cout << "\nCompile-Time Degree (double, N=6, iters=1):\n";
    std::cout << "Poly eval at 0.0: " << poly_compiletime_d_default_iters(0.0) << std::endl;
    std::cout << "Actual at 0.0:    " << my_func_double(0.0) << std::endl;
    std::cout << "Coefficients count: " << poly_compiletime_d_default_iters.coeffs().size() << std::endl;

    // Call the helper for error checking
    check_errors(my_func_double, poly_compiletime_d_default_iters, domain_a3, domain_b3,
                 "my_func_double (N=6, iters=1)");

    // -----------------------------------------------------
    // 4. Compile-Time Degree (N), Custom Compile-Time Iterations
    // -----------------------------------------------------
    double domain_a4 = -2.0;
    double domain_b4 = 2.0;
    auto poly_compiletime_c_2iters = poly_eval::make_func_eval<8, 2>(my_func_complex, domain_a4, domain_b4);
    std::cout << "\nCompile-Time Degree (complex, N=8, iters=2):\n";
    std::cout << "Poly eval at 1.0: " << poly_compiletime_c_2iters(1.0) << std::endl;
    std::cout << "Actual at 1.0:    " << my_func_complex(1.0) << std::endl;
    std::cout << "Coefficients count: " << poly_compiletime_c_2iters.coeffs().size() << std::endl;

    // Call the helper for error checking
    check_errors(my_func_complex, poly_compiletime_c_2iters, domain_a4, domain_b4, "my_func_complex (N=8, iters=2)");

    return 0;
}