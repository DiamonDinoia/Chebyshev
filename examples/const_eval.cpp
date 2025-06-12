#include "fast_eval.hpp" // Assuming your poly_eval.h is renamed to fast_eval.h

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>     // For std::cos, std::sin, std::abs
#include <iomanip>   // For std::scientific, std::setprecision
#include <chrono>    // For timing

int main() {
  // Define some example functions
  auto my_func_double = [](double x) { return std::cos(2 * x); };
  auto my_func_float = [](float x) { return std::sin(x) + std::cos(x); };
  auto my_func_complex = [](double x) { return std::complex<double>(x * x, std::sin(x)); };

  // A simple constexpr function for the C++20 compile-time fitting test
  // NOTE: std::cos and std::sin are NOT constexpr until C++23.
  // For C++20, this must be a purely arithmetic function to work at compile-time.
  auto my_func_constexpr = [](double x) constexpr { return 2.0 * x * x * x - 3.0 * x + 1.0; };

  std::cout << "--- Timing Poly Evaluation APIs ---\n";

  // -------------------------------------------------------------------------
  // 1. Runtime Degree (n), Default Iterations (1)
  //    API: poly_eval::make_func_eval(Func F, int n, InputType a, InputType b)
  // -------------------------------------------------------------------------
  std::cout << "\n=== Runtime Fitting: Fixed Degree (n) ===\n";
  double domain_a1 = -0.5;
  double domain_b1 = 0.5;
  int n1 = 16;
  auto start_fit_1 = std::chrono::high_resolution_clock::now();
  auto poly_runtime_d_default_iters = poly_eval::make_func_eval(my_func_double, n1, domain_a1, domain_b1);
  auto end_fit_1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> fit_duration_1 = end_fit_1 - start_fit_1;

  std::cout << "Runtime Degree (double, n=" << n1 << ", iters=1):\n";
  std::cout << "  Fitting time: " << std::fixed << std::setprecision(6) << fit_duration_1.count() * 1000 << " ms\n";

  auto start_eval_1 = std::chrono::high_resolution_clock::now();
  volatile double eval_result_1 = poly_runtime_d_default_iters(0.0); // Use volatile to prevent optimization
  auto end_eval_1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> eval_duration_1 = end_eval_1 - start_eval_1;
  std::cout << "  Evaluation time (single point): " << std::fixed << std::setprecision(9) << eval_duration_1.count() * 1e6 << " us\n";

  // -------------------------------------------------------------------------
  // 2. Runtime Degree (n), Custom Compile-Time Iterations
  //    API: poly_eval::make_func_eval<Iters_compile_time>(Func F, int n, InputType a, InputType b)
  // -------------------------------------------------------------------------
  std::cout << "\n=== Runtime Fitting: Fixed Degree, Custom Iters ===\n";
  float domain_a2 = -static_cast<float>(M_PI);
  float domain_b2 = static_cast<float>(M_PI);
  int n2 = 8;
  constexpr size_t iters_float_custom = 3;
  auto start_fit_2 = std::chrono::high_resolution_clock::now();
  auto poly_runtime_f_3iters = poly_eval::make_func_eval<iters_float_custom>(my_func_float, n2, domain_a2, domain_b2);
  auto end_fit_2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> fit_duration_2 = end_fit_2 - start_fit_2;

  std::cout << "Runtime Degree (float, n=" << n2 << ", iters=" << iters_float_custom << "):\n";
  std::cout << "  Fitting time: " << std::fixed << std::setprecision(6) << fit_duration_2.count() * 1000 << " ms\n";

  auto start_eval_2 = std::chrono::high_resolution_clock::now();
  volatile float eval_result_2 = poly_runtime_f_3iters(0.0f);
  auto end_eval_2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> eval_duration_2 = end_eval_2 - start_eval_2;
  std::cout << "  Evaluation time (single point): " << std::fixed << std::setprecision(9) << eval_duration_2.count() * 1e6 << " us\n";


  // -------------------------------------------------------------------------
  // 3. Compile-Time Degree (N), Default Iterations (1)
  //    API: poly_eval::make_func_eval<N_compile_time>(Func F, InputType a, InputType b)
  // -------------------------------------------------------------------------
  std::cout << "\n=== Runtime Fitting: Compile-Time Fixed Degree ===\n";
  double domain_a3 = -5.0;
  double domain_b3 = 5.0;
  constexpr size_t N3 = 6;
  auto start_fit_3 = std::chrono::high_resolution_clock::now();
  auto poly_compiletime_d_default_iters = poly_eval::make_func_eval<N3>(my_func_double, domain_a3, domain_b3);
  auto end_fit_3 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> fit_duration_3 = end_fit_3 - start_fit_3;

  std::cout << "Compile-Time Degree (double, N=" << N3 << ", iters=1):\n";
  std::cout << "  Fitting time: " << std::fixed << std::setprecision(6) << fit_duration_3.count() * 1000 << " ms\n";

  auto start_eval_3 = std::chrono::high_resolution_clock::now();
  volatile double eval_result_3 = poly_compiletime_d_default_iters(0.0);
  auto end_eval_3 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> eval_duration_3 = end_eval_3 - start_eval_3;
  std::cout << "  Evaluation time (single point): " << std::fixed << std::setprecision(9) << eval_duration_3.count() * 1e6 << " us\n";


  // -------------------------------------------------------------------------
  // 4. Runtime Epsilon (C++17 API)
  //    API: poly_eval::make_func_eval<MaxN_val, NumEvalPoints_val, Iters_compile_time>(Func F, double eps, InputType a, InputType b)
  // -------------------------------------------------------------------------
  std::cout << "\n=== Runtime Fitting: Error-Driven (C++17-style) ===\n";
  double eps_runtime = 1e-10;
  constexpr size_t max_n_runtime_c17 = 32;
  constexpr size_t eval_points_runtime_c17 = 100;
  constexpr size_t iters_runtime_c17 = 2;

  double domain_a4 = -1.0;
  double domain_b4 = 1.0;
  auto start_fit_4 = std::chrono::high_resolution_clock::now();
  auto poly_error_driven_runtime_eps = poly_eval::make_func_eval<max_n_runtime_c17, eval_points_runtime_c17, iters_runtime_c17>(
      my_func_double, eps_runtime, domain_a4, domain_b4);
  auto end_fit_4 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> fit_duration_4 = end_fit_4 - start_fit_4;

  std::cout << "Error-Driven (double, runtime eps=" << eps_runtime << ", MaxN=" << max_n_runtime_c17
            << ", EvalPoints=" << eval_points_runtime_c17 << ", Iters=" << iters_runtime_c17 << "):\n";
  std::cout << "  Fitting time: " << std::fixed << std::setprecision(6) << fit_duration_4.count() * 1000 << " ms\n";

  auto start_eval_4 = std::chrono::high_resolution_clock::now();
  volatile double eval_result_4 = poly_error_driven_runtime_eps(0.5);
  auto end_eval_4 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> eval_duration_4 = end_eval_4 - start_eval_4;
  std::cout << "  Evaluation time (single point): " << std::fixed << std::setprecision(9) << eval_duration_4.count() * 1e6 << " us\n";


  // -------------------------------------------------------------------------
  // 5. Compile-Time Epsilon (C++20 API - Still runtime fitting due to FuncEval internal)
  //    API: poly_eval::make_func_eval<eps_val, MaxN_val, NumEvalPoints_val, Iters_compile_time>(Func F, InputType a, InputType b)
  // -------------------------------------------------------------------------
#if __cplusplus >= 202002L
  std::cout << "\n=== Runtime Fitting: Error-Driven (C++20-style, compile-time eps) ===\n";
  constexpr double COMPILE_TIME_EPS = 1e-9;
  constexpr size_t COMPILE_TIME_MAX_N = 40;
  constexpr size_t COMPILE_TIME_EVAL_POINTS = 120;
  constexpr size_t COMPILE_TIME_ITERS = 1;

  double domain_a5 = -4.0;
  double domain_b5 = 4.0;
  auto start_fit_5 = std::chrono::high_resolution_clock::now();
  auto poly_error_driven_compile_time_eps = poly_eval::make_func_eval<
      COMPILE_TIME_EPS, COMPILE_TIME_MAX_N, COMPILE_TIME_EVAL_POINTS, COMPILE_TIME_ITERS>(
      my_func_double, domain_a5, domain_b5);
  auto end_fit_5 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> fit_duration_5 = end_fit_5 - start_fit_5;

  std::cout << "Error-Driven (double, compile-time eps=" << COMPILE_TIME_EPS << ", MaxN=" << COMPILE_TIME_MAX_N
            << ", EvalPoints=" << COMPILE_TIME_EVAL_POINTS << ", Iters=" << COMPILE_TIME_ITERS << " - C++20+):\n";
  std::cout << "  Fitting time: " << std::fixed << std::setprecision(6) << fit_duration_5.count() * 1000 << " ms\n";

  auto start_eval_5 = std::chrono::high_resolution_clock::now();
  volatile double eval_result_5 = poly_error_driven_compile_time_eps(0.1);
  auto end_eval_5 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> eval_duration_5 = end_eval_5 - start_eval_5;
  std::cout << "  Evaluation time (single point): " << std::fixed << std::setprecision(9) << eval_duration_5.count() * 1e6 << " us\n";

#else
  std::cout << "\n(Skipping C++20 compile-time epsilon error-driven test: C++20 or later not enabled)\n";
#endif


  // -------------------------------------------------------------------------
  // 6. Full Compile-Time Fitting and Evaluation (C++20 API - make_constexpr_fixed_degree_eval)
  //    API: poly_eval::make_constexpr_fixed_degree_eval<N_DEGREE, Iters_compile_time>(Func F, InputType a, InputType b)
  // -------------------------------------------------------------------------
#if __cplusplus >= 202002L
  std::cout << "\n=== Full Compile-Time Fitting and Evaluation (C++20+) ===\n";
  constexpr size_t CONSTEXPR_DEGREE = 5; // Fixed degree for compile-time fitting
  constexpr size_t CONSTEXPR_ITERS = 2; // Fixed iterations for compile-time fitting

  constexpr double domain_a6 = -1.0;
  constexpr double domain_b6 = 1.0;

  // --- Compile-Time Fitting ---
  // The 'fitting' for poly_constexpr_eval happens *at compile time*.
  // There is no runtime cost associated with this fitting. The coefficients
  // are pre-computed and stored directly in the executable's data segment.
  std::cout << "Full Compile-Time (double, Fixed N=" << CONSTEXPR_DEGREE
            << ", Iters=" << CONSTEXPR_ITERS << " - C++20+):\n";
  std::cout << "  Fitting time: 0.000000 ms (performed at compile-time)\n";
  constexpr auto poly_constexpr_eval = poly_eval::make_func_eval<
      CONSTEXPR_DEGREE, CONSTEXPR_ITERS>(my_func_constexpr, domain_a6, domain_b6);

  // --- Compile-Time Evaluation (if the point is also constexpr) ---
  constexpr double test_pt_constexpr = 0.5;
  // This evaluation also happens at compile-time, resulting in zero runtime cost.
  constexpr double eval_result_constexpr = poly_constexpr_eval(test_pt_constexpr);
  std::cout << "  Evaluation time (single point, constexpr): 0.000000000 us (performed at compile-time)\n";


  // --- Runtime Evaluation ---
  // Measure runtime evaluation performance of the compile-time fitted polynomial
  auto start_eval_6 = std::chrono::high_resolution_clock::now();
  volatile double eval_result_6 = poly_constexpr_eval(0.7); // Evaluate at a runtime point
  auto end_eval_6 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> eval_duration_6 = end_eval_6 - start_eval_6;
  std::cout << "  Evaluation time (single point, runtime): " << std::fixed << std::setprecision(9) << eval_duration_6.count() * 1e6 << " us\n";

#else
  std::cout << "\n(Skipping Full Compile-Time Fitting test: C++20 or later not enabled)\n";
#endif

  return 0;
}