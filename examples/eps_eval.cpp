#include "fast_eval.hpp" // Assuming your poly_eval.h is renamed to fast_eval.h

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>     // For std::cos, std::sin, std::abs
#include <iomanip>   // For std::scientific, std::setprecision, std::setw, std::left, std::right
#include <random>    // For std::random_device, std::mt19937, std::uniform_real_distribution
#include <limits>    // For std::numeric_limits


// Helper function to perform and print error checks in a table format
template <typename TFunc, typename TPoly, typename TInput>
void check_errors(TFunc original_func, TPoly poly_evaluator,
                  TInput domain_a, TInput Input_b, const std::string &description) {
  // Explicitly deduce TOutput from the TPoly's OutputType
  using TOutput = typename TPoly::OutputType;

  std::cout << "\n--- Relative Error Check for " << description
      << " on [" << domain_a << ", " << Input_b << "] ---\n";

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
  std::cout << std::setw(ERROR_WIDTH) << std::left << "Relative Error (1-|P/A|)";
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
  // Define some example functions
  auto my_func_double = [](double x) { return std::cos(2 * x); };
  auto my_func_float = [](float x) { return std::sin(x) + std::cos(x); };
  auto my_func_complex = [](double x) { return std::complex<double>(x * x, std::sin(x)); };

  std::cout << "--- Testing FuncEval with make_func_eval (Error-Driven) API ---\n";

  // -----------------------------------------------------
  // 1. Error-Driven Degree for double function (RUNTIME eps, C++17 friendly)
  //    MaxN_val=64, NumEvalPoints_val=100 are compile-time constants
  // -----------------------------------------------------
  double eps_double_runtime = 1e-15; // Target error (runtime value)
  constexpr size_t MAX_N_DOUBLE_C17 = 64; // Max degree to search (compile-time)
  constexpr size_t EVAL_POINTS_DOUBLE_C17 = 100; // Number of evaluation points (compile-time)

  double domain_a1 = -.4;
  double domain_b1 = .3;
  auto poly_runtime_d_eps = poly_eval::make_func_eval<MAX_N_DOUBLE_C17, EVAL_POINTS_DOUBLE_C17>(
      my_func_double, eps_double_runtime, domain_a1, domain_b1);
  std::cout << "\nError-Driven (double, RUNTIME Epsilon=" << std::scientific << std::setprecision(2) <<
      eps_double_runtime
      << ", MaxN=" << MAX_N_DOUBLE_C17 << ", EvalPoints=" << EVAL_POINTS_DOUBLE_C17 << " - C++17+):\n";
  std::cout << "  Actual degree found: " << poly_runtime_d_eps.coeffs().size() << std::endl;
  std::cout << "  Poly eval at 0.0: " << poly_runtime_d_eps(0.0) << std::endl;
  std::cout << "  Actual at 0.0:    " << my_func_double(0.0) << std::endl;
  check_errors(my_func_double, poly_runtime_d_eps,
               domain_a1, domain_b1, "my_func_double (runtime eps, compile-time max_n, eval_pts)");

  // -----------------------------------------------------
  // 2. Error-Driven Degree for float function (RUNTIME eps with Custom Iters, C++17 friendly)
  //    MaxN_val=32, NumEvalPoints_val=100 are compile-time constants
  // -----------------------------------------------------
  float eps_float_runtime = 1e-6f; // Runtime value
  constexpr size_t MAX_N_FLOAT_C17 = 32; // Compile-time
  constexpr size_t EVAL_POINTS_FLOAT_C17 = 100; // Compile-time
  constexpr size_t ITERS_FLOAT_C17 = 3; // Compile-time refinement iterations

  float domain_a2 = -static_cast<float>(0);
  float domain_b2 = static_cast<float>(1);
  auto poly_runtime_f_eps = poly_eval::make_func_eval<MAX_N_FLOAT_C17, EVAL_POINTS_FLOAT_C17, ITERS_FLOAT_C17>(
      my_func_float, eps_float_runtime, domain_a2, domain_b2);
  std::cout << "\nError-Driven (float, RUNTIME Epsilon=" << std::scientific << std::setprecision(2) << eps_float_runtime
      << ", MaxN=" << MAX_N_FLOAT_C17 << ", EvalPoints=" << EVAL_POINTS_FLOAT_C17
      << ", Iters=" << ITERS_FLOAT_C17 << " - C++17+):\n";
  std::cout << "  Actual degree found: " << poly_runtime_f_eps.coeffs().size() << std::endl;
  std::cout << "  Poly eval at 0.0f: " << poly_runtime_f_eps(0.0f) << std::endl;
  std::cout << "  Actual at 0.0f:    " << my_func_float(0.0f) << std::endl;
  check_errors(my_func_float, poly_runtime_f_eps,
               domain_a2, domain_b2, "my_func_float (runtime eps, compile-time max_n, eval_pts, iters)");

  // -----------------------------------------------------
  // 3. Error-Driven Degree for complex function (RUNTIME eps, C++17 friendly)
  //    MaxN_val=48, NumEvalPoints_val=150 are compile-time constants
  // -----------------------------------------------------
  double eps_complex_runtime = 1e-8; // Runtime value
  constexpr size_t MAX_N_COMPLEX_C17 = 48; // Compile-time
  constexpr size_t EVAL_POINTS_COMPLEX_C17 = 150; // Compile-time

  double domain_a3 = -2.0;
  double domain_b3 = 2.0;
  auto poly_runtime_c_eps = poly_eval::make_func_eval<MAX_N_COMPLEX_C17, EVAL_POINTS_COMPLEX_C17>(
      my_func_complex, eps_complex_runtime, domain_a3, domain_b3);
  std::cout << "\nError-Driven (complex, RUNTIME Epsilon=" << std::scientific << std::setprecision(2) <<
      eps_complex_runtime
      << ", MaxN=" << MAX_N_COMPLEX_C17 << ", EvalPoints=" << EVAL_POINTS_COMPLEX_C17 << " - C++17+):\n";
  std::cout << "  Actual degree found: " << poly_runtime_c_eps.coeffs().size() << std::endl;
  std::cout << "  Poly eval at 1.0: " << poly_runtime_c_eps(1.0) << std::endl;
  std::cout << "  Actual at 1.0:    " << my_func_complex(1.0) << std::endl;
  check_errors(my_func_complex, poly_runtime_c_eps,
               domain_a3, domain_b3, "my_func_complex (runtime eps, compile-time max_n, eval_pts)");

  // -----------------------------------------------------
  // 4. Error-Driven Degree for double function (COMPILE-TIME eps, max_n, num_eval_points - C++20 only)
  // -----------------------------------------------------
#if __cplusplus >= 202002L
  // Note: All numeric parameters are now template arguments, so they must be compile-time literals
  constexpr double COMPILE_TIME_EPS = 1e-9;
  constexpr size_t COMPILE_TIME_MAX_N = 64;
  constexpr size_t COMPILE_TIME_EVAL_POINTS = 200;

  double domain_a4 = -3.0;
  double domain_b4 = 3.0;
  auto poly_compiletime_d_eps = poly_eval::make_func_eval<
    COMPILE_TIME_EPS, COMPILE_TIME_MAX_N, COMPILE_TIME_EVAL_POINTS>(my_func_double, domain_a4, domain_b4);
  std::cout << "\nError-Driven (double, COMPILE-TIME Epsilon=" << std::scientific << std::setprecision(2) <<
      COMPILE_TIME_EPS
      << ", MaxN=" << COMPILE_TIME_MAX_N << ", EvalPoints=" << COMPILE_TIME_EVAL_POINTS << " - C++20+):\n";
  std::cout << "  Actual degree found: " << poly_compiletime_d_eps.coeffs().size() << std::endl;
  std::cout << "  Poly eval at 0.0: " << poly_compiletime_d_eps(0.0) << std::endl;
  std::cout << "  Actual at 0.0:    " << my_func_double(0.0) << std::endl;
  check_errors(my_func_double, poly_compiletime_d_eps,
               domain_a4, domain_b4, "my_func_double (compile-time eps, max_n, eval_pts)");
#else
    std::cout << "\n(Skipping C++20 compile-time epsilon, max_n, num_eval_points test: C++20 or later not enabled)\n";
#endif

  return 0;
}