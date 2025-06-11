#include <iostream>
#include <vector>
#include <cmath>     // For std::fabs, std::pow, std::fma, std::isnan, std::isinf
#include <array>     // For std::array for coefficients
#include <chrono>    // For benchmarking
#include <string>    // For std::string in Benchmark
#include <tuple>     // For std::make_tuple, std::get
#include <utility>   // For std::index_sequence, std::make_index_sequence
#include <optional>  // For std::optional
#include <iomanip>   // For std::setw, std::left, std::right, std::fixed, std::setprecision
#include <random>    // For random number generation
#include <algorithm> // For std::generate

// --- Common Helper Function ---
// __always_inline is a GCC/Clang extension. For MSVC, use __forceinline.
// For standard C++, rely on 'inline' or compiler's optimization capabilities.
template <typename T>
#ifdef __GNUC__
__attribute__((always_inline))
#elif _MSC_VER
__forceinline
#endif
constexpr T fma(T a, T b, T c) {
  return std::fma(a, b, c);
}

// --- Compile-time Log2 function ---
// Computes log2(N) at compile time. Used to determine powers of x for Estrin's scheme.
template <std::size_t N>
#ifdef __GNUC__
__attribute__((always_inline))
#elif _MSC_VER
__forceinline
#endif
constexpr std::size_t ct_log2() {
  static_assert(N > 0, "ct_log2 of zero is undefined.");
  if constexpr (N == 1) {
    return 0; // log2(1) = 0
  } else {
    // Recursive call to find the highest bit set
    return ct_log2<(N / 2)>() + 1;
  }
}

// --- Helper to compute x^(2^k) at compile time for building the initial powers tuple ---
// This function recursively calculates x^(2^K) values at compile time.
template <typename T, std::size_t K>
#ifdef __GNUC__
__attribute__((always_inline))
#elif _MSC_VER
__forceinline
#endif
constexpr T compute_x_power_of_two(T x) {
  if constexpr (K == 0) {
    return x; // x^(2^0) = x^1
  } else {
    T prev_power = compute_x_power_of_two<T, K - 1>(x);
    return prev_power * prev_power; // x^(2^K) = (x^(2^(K-1)))^2
  }
}

// --- Helper to create tuple of powers for Estrin's scheme ---
// Uses std::index_sequence to generate a tuple of powers (x^1, x^2, x^4, x^8, ...)
template <typename T, std::size_t... K>
#ifdef __GNUC__
__attribute__((always_inline))
#elif _MSC_VER
__forceinline
#endif
constexpr auto make_powers_tuple(T val, std::index_sequence<K...>) {
  return std::make_tuple(compute_x_power_of_two<T, K>(val)...);
}

// --- Estrin's Scheme Recursive Evaluation ---
// The core recursive function for Estrin's polynomial evaluation.
// Uses if constexpr to unroll the recursion at compile time, leading to highly optimized code.
template <typename T, typename PowersTuple, std::size_t Count>
#ifdef __GNUC__
__attribute__((always_inline))
#elif _MSC_VER
__forceinline
#endif
constexpr T estrin_eval_recursive(
    T x, PowersTuple powers_tuple, const T *coeffs) {
  if constexpr (Count == 0) {
    return T{}; // Should not be reached with N_coeffs > 0 and Count > 0
  } else if constexpr (Count == 1) {
    return coeffs[0]; // Degree 0: P(x) = c0
  } else {
    // Determine the split point based on the largest power of 2 less than or equal to current_degree.
    constexpr std::size_t current_degree = Count - 1;
    constexpr std::size_t split_power_of_2 =
        (current_degree == 0) ? 1 : (1ULL << ct_log2<current_degree>());

    // The index into the powers_tuple corresponds to log2(split_power_of_2).
    constexpr std::size_t x_power_log2_idx = ct_log2<split_power_of_2>();

    // Apply Estrin's formula: P(x) = P_lower(x) + x^P2 * P_higher(x)
    return fma(std::get<x_power_log2_idx>(powers_tuple),
               estrin_eval_recursive<T, PowersTuple, Count - split_power_of_2>(
                   x, powers_tuple, coeffs + split_power_of_2),
               estrin_eval_recursive<T, PowersTuple, split_power_of_2>(x, powers_tuple, coeffs));
  }
}

// --- Public Interface for Estrin's Polynomial Evaluation ---
// This is the entry point for Estrin's scheme.
// It initializes the powers tuple and calls the recursive evaluation.
template <typename T, std::size_t N_coeffs>
#ifdef __GNUC__
__attribute__((always_inline))
#elif _MSC_VER
__forceinline
#endif
constexpr T estrin_eval(T x, const T *coeffs) {
  static_assert(N_coeffs > 0, "At least one coefficient (c0) is required for a polynomial.");

  if constexpr (N_coeffs == 1) {
    return coeffs[0]; // Special case for degree 0 (constant polynomial)
  } else {
    constexpr std::size_t max_degree = N_coeffs - 1;
    // Calculate the maximum 'k' needed for x^(2^k) in the powers tuple
    // (e.g., for degree 15, max_degree is 15, ct_log2<15> is 3, so we need x^1, x^2, x^4, x^8)
    constexpr std::size_t max_k_value_for_powers_tuple = ct_log2<max_degree>();

    // Create a tuple of powers (x^1, x^2, x^4, ..., x^(2^max_k_value_for_powers_tuple))
    const auto powers_tuple = make_powers_tuple(x, std::make_index_sequence<max_k_value_for_powers_tuple + 1>{});

    // Start the recursive evaluation
    return estrin_eval_recursive<T, decltype(powers_tuple), N_coeffs>(x, powers_tuple, coeffs);
  }
}

// --- Unrolled Estrin's Macro-like Functions (for degrees 0-15) ---
// These are explicit, manually unrolled implementations for specific low degrees,
// often found in highly optimized numerical libraries like SLEEF. They assume
// powers of x (x, x2, x4, x8) are pre-computed.
template <typename T>
#ifdef __GNUC__
__attribute__((always_inline))
#elif _MSC_VER
__forceinline
#endif
constexpr T sleef_poly_eval_0(T /*x*/, const T *coeffs) {
  // N_coeffs = 1 (degree 0)
  return coeffs[0];
}

template <typename T>
#ifdef __GNUC__
__attribute__((always_inline))
#elif _MSC_VER
__forceinline
#endif
constexpr T sleef_poly_eval_1(T x, const T *coeffs) {
  // N_coeffs = 2 (degree 1)
  return fma(x, coeffs[1], coeffs[0]);
}

template <typename T>
#ifdef __GNUC__
__attribute__((always_inline))
#elif _MSC_VER
__forceinline
#endif
constexpr T sleef_poly_eval_2(T x, T x2, const T *coeffs) {
  // N_coeffs = 3 (degree 2)
  return fma(x2, coeffs[2], fma(x, coeffs[1], coeffs[0]));
}

template <typename T>
#ifdef __GNUC__
__attribute__((always_inline))
#elif _MSC_VER
__forceinline
#endif
constexpr T sleef_poly_eval_3(T x, T x2, const T *coeffs) {
  // N_coeffs = 4 (degree 3)
  return fma(x2, fma(x, coeffs[3], coeffs[2]), fma(x, coeffs[1], coeffs[0]));
}

template <typename T>
#ifdef __GNUC__
__attribute__((always_inline))
#elif _MSC_VER
__forceinline
#endif
constexpr T sleef_poly_eval_4(T x, T x2, T x4, const T *coeffs) {
  // N_coeffs = 5 (degree 4)
  return fma(x4, coeffs[4], sleef_poly_eval_3(x, x2, coeffs));
}

template <typename T>
#ifdef __GNUC__
__attribute__((always_inline))
#elif _MSC_VER
__forceinline
#endif
constexpr T sleef_poly_eval_5(T x, T x2, T x4, const T *coeffs) {
  // N_coeffs = 6 (degree 5)
  return fma(x4, sleef_poly_eval_1(x, coeffs + 4), sleef_poly_eval_3(x, x2, coeffs));
}

template <typename T>
#ifdef __GNUC__
__attribute__((always_inline))
#elif _MSC_VER
__forceinline
#endif
constexpr T sleef_poly_eval_6(T x, T x2, T x4, const T *coeffs) {
  // N_coeffs = 7 (degree 6)
  return fma(x4, sleef_poly_eval_2(x, x2, coeffs + 4), sleef_poly_eval_3(x, x2, coeffs));
}

template <typename T>
#ifdef __GNUC__
__attribute__((always_inline))
#elif _MSC_VER
__forceinline
#endif
constexpr T sleef_poly_eval_7(T x, T x2, T x4, const T *coeffs) {
  // N_coeffs = 8 (degree 7)
  return fma(x4, sleef_poly_eval_3(x, x2, coeffs + 4), sleef_poly_eval_3(x, x2, coeffs));
}

template <typename T>
#ifdef __GNUC__
__attribute__((always_inline))
#elif _MSC_VER
__forceinline
#endif
constexpr T sleef_poly_eval_8(T x, T x2, T x4, T x8, const T *coeffs) {
  // N_coeffs = 9 (degree 8)
  return fma(x8, coeffs[8], sleef_poly_eval_7(x, x2, x4, coeffs));
}

template <typename T>
#ifdef __GNUC__
__attribute__((always_inline))
#elif _MSC_VER
__forceinline
#endif
constexpr T sleef_poly_eval_9(T x, T x2, T x4, T x8, const T *coeffs) {
  // N_coeffs = 10 (degree 9)
  return fma(x8, sleef_poly_eval_1(x, coeffs + 8), sleef_poly_eval_7(x, x2, x4, coeffs));
}

template <typename T>
#ifdef __GNUC__
__attribute__((always_inline))
#elif _MSC_VER
__forceinline
#endif
constexpr T sleef_poly_eval_10(T x, T x2, T x4, T x8, const T *coeffs) {
  // N_coeffs = 11 (degree 10)
  return fma(x8, sleef_poly_eval_2(x, x2, coeffs + 8), sleef_poly_eval_7(x, x2, x4, coeffs));
}

template <typename T>
#ifdef __GNUC__
__attribute__((always_inline))
#elif _MSC_VER
__forceinline
#endif
constexpr T sleef_poly_eval_11(T x, T x2, T x4, T x8, const T *coeffs) {
  // N_coeffs = 12 (degree 11)
  return fma(x8, sleef_poly_eval_3(x, x2, coeffs + 8), sleef_poly_eval_7(x, x2, x4, coeffs));
}

template <typename T>
#ifdef __GNUC__
__attribute__((always_inline))
#elif _MSC_VER
__forceinline
#endif
constexpr T sleef_poly_eval_12(T x, T x2, T x4, T x8, const T *coeffs) {
  // N_coeffs = 13 (degree 12)
  return fma(x8, sleef_poly_eval_4(x, x2, x4, coeffs + 8), sleef_poly_eval_7(x, x2, x4, coeffs));
}

template <typename T>
#ifdef __GNUC__
__attribute__((always_inline))
#elif _MSC_VER
__forceinline
#endif
constexpr T sleef_poly_eval_13(T x, T x2, T x4, T x8, const T *coeffs) {
  // N_coeffs = 14 (degree 13)
  return fma(x8, sleef_poly_eval_5(x, x2, x4, coeffs + 8), sleef_poly_eval_7(x, x2, x4, coeffs));
}

template <typename T>
#ifdef __GNUC__
__attribute__((always_inline))
#elif _MSC_VER
__forceinline
#endif
constexpr T sleef_poly_eval_14(T x, T x2, T x4, T x8, const T *coeffs) {
  // N_coeffs = 15 (degree 14)
  return fma(x8, sleef_poly_eval_6(x, x2, x4, coeffs + 8), sleef_poly_eval_7(x, x2, x4, coeffs));
}

template <typename T>
#ifdef __GNUC__
__attribute__((always_inline))
#elif _MSC_VER
__forceinline
#endif
constexpr T sleef_poly_eval_15(T x, T x2, T x4, T x8, const T *coeffs) {
  // N_coeffs = 16 (degree 15)
  return fma(x8, sleef_poly_eval_7(x, x2, x4, coeffs + 8), sleef_poly_eval_7(x, x2, x4, coeffs));
}

// --- General Purpose Horner's Method (pointer and compile-time size version) ---
// A standard and numerically stable method for polynomial evaluation.
template <typename T, std::size_t N_coeffs>
#ifdef __GNUC__
__attribute__((always_inline))
#elif _MSC_VER
__forceinline
#endif
constexpr T horner_eval(T x, const T *coeffs) {
  static_assert(N_coeffs > 0, "At least one coefficient is required for a polynomial.");
  T result = coeffs[N_coeffs - 1]; // Start with the highest degree coefficient
  if constexpr (N_coeffs > 1) {
    // Iterate downwards, applying Horner's rule: P(x) = c0 + x(c1 + x(c2 + ...))
    for (std::size_t i = N_coeffs - 1; i-- > 0;) {
      result = fma(x, result, coeffs[i]);
    }
  }
  return result;
}

// --- Benchmark Utilities ---
// Helper struct to measure execution time.
struct BenchmarkTimer {
  std::string name;
  std::chrono::high_resolution_clock::time_point start_time;

  // __attribute_noinline__ prevents the compiler from inlining this function,
  // ensuring the timer overhead is measured consistently.
  #ifdef __GNUC__
  __attribute__((noinline))
  #elif _MSC_VER
  __declspec(noinline)
  #endif
  BenchmarkTimer(const std::string &n) : name(n) {
    start_time = std::chrono::high_resolution_clock::now();
  }

  #ifdef __GNUC__
  __attribute__((noinline))
  #elif _MSC_VER
  __declspec(noinline)
  #endif
  long long stop() {
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
  }
};

// Struct to hold benchmark results for a single polynomial degree.
struct DegreeBenchmarkResults {
  std::size_t degree;
  double estrin_ns_per_eval;
  double horner_ns_per_eval;
  std::optional<double> sleef_poly_eval_ns_per_eval;
};

// Static template class to run and collect benchmarks for specified degrees.
template <std::size_t... Degrees>
struct PolynomialBenchmarkSuite {
  static inline std::vector<DegreeBenchmarkResults> all_results; // Stores results from all degrees

  // Runs all benchmarks for the specified degrees.
  static void run_all(double x_val, const double *coeffs_ptr, int num_iterations) {
    std::cout << "\n--- Benchmarking Polynomials (Degree 0 to 32) ---\n";
    std::cout << "x for performance test = " << x_val << ", Iterations per benchmark = " << num_iterations << "\n\n";

    // Unpack the Degrees template parameter pack to run benchmark for each degree.
    ((run_single_degree_benchmark<Degrees>(x_val, coeffs_ptr, num_iterations)), ...);

    print_summary_table();
  }

private:
  // Runs benchmarks for a single polynomial degree.
  template <std::size_t Degree>
  static void run_single_degree_benchmark(double x_val, const double *coeffs_ptr, int num_iterations) {
    constexpr std::size_t num_coeffs = Degree + 1; // Number of coefficients for this degree

    // Skip if not enough coefficients are provided for this degree.
    // Assuming coeffs_ptr points to an array of MAX_COEFFS (33 for deg 32).
    if (num_coeffs > 33) {
        std::cout << "Skipping Degree " << Degree << ": Not enough coefficients provided in the coeffs_array.\n";
        return;
    }

    DegreeBenchmarkResults current_degree_results;
    current_degree_results.degree = Degree;

    std::cout << "--- Degree " << Degree << " (Coeffs: " << num_coeffs << ") ---\n";

    // --- Performance Benchmarks ---
    // Estrin's Benchmark
    volatile double sum_estrin = 0.0; // volatile to prevent optimization of the sum
    long long estrin_total_ns;
    {
      BenchmarkTimer timer("Estrin's (Deg " + std::to_string(Degree) + ")");
      for (int i = 0; i < num_iterations; ++i) {
        volatile double x = x_val; // volatile to ensure x_val is re-read or optimized as needed
        sum_estrin += estrin_eval<double, num_coeffs>(x, coeffs_ptr);
      }
      estrin_total_ns = timer.stop();
      current_degree_results.estrin_ns_per_eval = static_cast<double>(estrin_total_ns) / num_iterations;
    }
    std::cout << "  Estrin Perf Sum: " << sum_estrin << "\n";
    std::cout << "  Estrin's: " << std::fixed << std::setprecision(2) << current_degree_results.estrin_ns_per_eval <<
        " ns/eval\n";

    // Horner's (Ptr) Benchmark
    volatile double sum_horner = 0.0;
    long long horner_total_ns;
    {
      BenchmarkTimer timer("Horner's (Ptr, Deg " + std::to_string(Degree) + ")");
      for (int i = 0; i < num_iterations; ++i) {
        volatile double x = x_val;
        sum_horner += horner_eval<double, num_coeffs>(x, coeffs_ptr);
      }
      horner_total_ns = timer.stop();
      current_degree_results.horner_ns_per_eval = static_cast<double>(horner_total_ns) / num_iterations;
    }
    std::cout << "  Horner (Ptr) Perf Sum: " << sum_horner << "\n";
    std::cout << "  Horner's (Ptr): " << std::fixed << std::setprecision(2) << current_degree_results.
        horner_ns_per_eval << " ns/eval\n";

    // Unrolled Estrin Macro Functions Benchmark (using if constexpr)
    volatile double sum_sleef_poly = 0.0;
    bool sleef_poly_applicable = true;
    long long sleef_poly_total_ns = 0;
    {
      BenchmarkTimer timer("Unrolled Estrin (Deg " + std::to_string(Degree) + ")");
      for (int i = 0; i < num_iterations; ++i) {
        // These powers must be volatile and computed inside the loop
        // to accurately reflect the overhead of computing them for each evaluation.
        volatile double x_local = x_val;
        volatile double x_power2 = x_local * x_local;
        volatile double x_power4 = x_power2 * x_power2;
        volatile double x_power8 = x_power4 * x_power4;

        // Use if constexpr to select the correct unrolled function at compile time.
        if constexpr (Degree == 0) {
          sum_sleef_poly += sleef_poly_eval_0(x_local, coeffs_ptr);
        } else if constexpr (Degree == 1) {
          sum_sleef_poly += sleef_poly_eval_1(x_local, coeffs_ptr);
        } else if constexpr (Degree == 2) {
          sum_sleef_poly += sleef_poly_eval_2(x_local, x_power2, coeffs_ptr);
        } else if constexpr (Degree == 3) {
          sum_sleef_poly += sleef_poly_eval_3(x_local, x_power2, coeffs_ptr);
        } else if constexpr (Degree == 4) {
          sum_sleef_poly += sleef_poly_eval_4(x_local, x_power2, x_power4, coeffs_ptr);
        } else if constexpr (Degree == 5) {
          sum_sleef_poly += sleef_poly_eval_5(x_local, x_power2, x_power4, coeffs_ptr);
        } else if constexpr (Degree == 6) {
          sum_sleef_poly += sleef_poly_eval_6(x_local, x_power2, x_power4, coeffs_ptr);
        } else if constexpr (Degree == 7) {
          sum_sleef_poly += sleef_poly_eval_7(x_local, x_power2, x_power4, coeffs_ptr);
        } else if constexpr (Degree == 8) {
          sum_sleef_poly += sleef_poly_eval_8(x_local, x_power2, x_power4, x_power8, coeffs_ptr);
        } else if constexpr (Degree == 9) {
          sum_sleef_poly += sleef_poly_eval_9(x_local, x_power2, x_power4, x_power8, coeffs_ptr);
        } else if constexpr (Degree == 10) {
          sum_sleef_poly += sleef_poly_eval_10(x_local, x_power2, x_power4, x_power8, coeffs_ptr);
        } else if constexpr (Degree == 11) {
          sum_sleef_poly += sleef_poly_eval_11(x_local, x_power2, x_power4, x_power8, coeffs_ptr);
        } else if constexpr (Degree == 12) {
          sum_sleef_poly += sleef_poly_eval_12(x_local, x_power2, x_power4, x_power8, coeffs_ptr);
        } else if constexpr (Degree == 13) {
          sum_sleef_poly += sleef_poly_eval_13(x_local, x_power2, x_power4, x_power8, coeffs_ptr);
        } else if constexpr (Degree == 14) {
          sum_sleef_poly += sleef_poly_eval_14(x_local, x_power2, x_power4, x_power8, coeffs_ptr);
        } else if constexpr (Degree == 15) {
          sum_sleef_poly += sleef_poly_eval_15(x_local, x_power2, x_power4, x_power8, coeffs_ptr);
        } else {
          sleef_poly_applicable = false; // Macro not defined for degrees > 15
        }
      }
      sleef_poly_total_ns = timer.stop();
      if (sleef_poly_applicable) {
        current_degree_results.sleef_poly_eval_ns_per_eval = static_cast<double>(sleef_poly_total_ns) / num_iterations;
        std::cout << "  Unrolled Estrin Perf Sum: " << sum_sleef_poly << "\n";
        std::cout << "  Unrolled Estrin: " << std::fixed << std::setprecision(2) << *current_degree_results.
            sleef_poly_eval_ns_per_eval << " ns/eval\n";
      } else {
        current_degree_results.sleef_poly_eval_ns_per_eval = std::nullopt;
        std::cout << "  Unrolled Estrin Perf Sum: N/A (No specific unrolled macro for degree " << Degree << ")\n";
        std::cout << "  Unrolled Estrin: N/A\n";
      }
    }

    std::cout << "\n"; // Add extra newline for spacing

    // --- Accuracy Verification ---
    // Lambda for robust verification, handling near-zero expected values and NaNs/Infs.
    auto check_accuracy = [&](const std::string &name_suffix, double actual_val, double expected_val, double test_x) {
      // Save current precision setting
      std::streamsize original_precision = std::cout.precision();

      if (std::isnan(actual_val) || std::isinf(actual_val)) {
          std::cout << "    SKIPPED: " << name_suffix << " (result for x=" << test_x << " is NaN/Inf).\n";
      } else {
          // Define tolerances
          const double abs_tolerance = 1e-14; // A small absolute tolerance for values near zero
          const double rel_tolerance = 1e-12;  // A relative tolerance for larger values (adjust as needed for precision)

          // Combined tolerance: max(absolute_tolerance, relative_tolerance * abs(expected))
          // This handles cases where expected is small (uses abs_tolerance) or large (uses rel_tolerance)
          double combined_tolerance = std::max(abs_tolerance, rel_tolerance * std::abs(expected_val));
          double absolute_diff = std::abs(actual_val - expected_val);

          if (absolute_diff <= combined_tolerance) {
              std::cout << "    OK: " << name_suffix << " for x=" << test_x << " is very close.\n";
          } else {
              std::cout << "    FAILED: " << name_suffix << " for x=" << test_x << " mismatch.\n";
              // Print full precision for debugging actual differences
              std::cout << std::fixed << std::setprecision(20);
              std::cout << "      Actual: " << actual_val << "\n";
              std::cout << "      Expected: " << expected_val << "\n";
              std::cout << "      Absolute diff: " << absolute_diff << "\n";
              if (std::abs(expected_val) > 1e-300) { // Avoid division by zero for extremely small expected values
                  std::cout << "      Relative diff: |1 - (Actual / Expected)| = " << std::abs(1 - actual_val / expected_val) << "\n";
              }
          }
      }
      // Restore original precision setting
      std::cout << std::fixed << std::setprecision(original_precision);
    };

    std::cout << "--- Accuracy Test for Degree " << Degree << " ---\n";
    const std::vector<double> accuracy_x_values = {-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0};

    for (double test_x : accuracy_x_values) {
        double expected_result = 0.0;
        for (std::size_t i = 0; i < num_coeffs; ++i) {
            expected_result += coeffs_ptr[i] * std::pow(test_x, i);
        }

        // Test Estrin's accuracy
        double estrin_actual = estrin_eval<double, num_coeffs>(test_x, coeffs_ptr);
        check_accuracy("Estrin's", estrin_actual, expected_result, test_x);

        // Test Horner's accuracy
        double horner_actual = horner_eval<double, num_coeffs>(test_x, coeffs_ptr);
        check_accuracy("Horner's (Ptr)", horner_actual, expected_result, test_x);

        // Test Unrolled Estrin accuracy (if applicable)
        if (sleef_poly_applicable) {
            volatile double x_local = test_x;
            volatile double x_power2 = x_local * x_local;
            volatile double x_power4 = x_power2 * x_power2;
            volatile double x_power8 = x_power4 * x_power4;
            double sleef_poly_actual = 0.0;

            if constexpr (Degree == 0) { sleef_poly_actual = sleef_poly_eval_0(x_local, coeffs_ptr); }
            else if constexpr (Degree == 1) { sleef_poly_actual = sleef_poly_eval_1(x_local, coeffs_ptr); }
            else if constexpr (Degree == 2) { sleef_poly_actual = sleef_poly_eval_2(x_local, x_power2, coeffs_ptr); }
            else if constexpr (Degree == 3) { sleef_poly_actual = sleef_poly_eval_3(x_local, x_power2, coeffs_ptr); }
            else if constexpr (Degree == 4) { sleef_poly_actual = sleef_poly_eval_4(x_local, x_power2, x_power4, coeffs_ptr); }
            else if constexpr (Degree == 5) { sleef_poly_actual = sleef_poly_eval_5(x_local, x_power2, x_power4, coeffs_ptr); }
            else if constexpr (Degree == 6) { sleef_poly_actual = sleef_poly_eval_6(x_local, x_power2, x_power4, coeffs_ptr); }
            else if constexpr (Degree == 7) { sleef_poly_actual = sleef_poly_eval_7(x_local, x_power2, x_power4, coeffs_ptr); }
            else if constexpr (Degree == 8) { sleef_poly_actual = sleef_poly_eval_8(x_local, x_power2, x_power4, x_power8, coeffs_ptr); }
            else if constexpr (Degree == 9) { sleef_poly_actual = sleef_poly_eval_9(x_local, x_power2, x_power4, x_power8, coeffs_ptr); }
            else if constexpr (Degree == 10) { sleef_poly_actual = sleef_poly_eval_10(x_local, x_power2, x_power4, x_power8, coeffs_ptr); }
            else if constexpr (Degree == 11) { sleef_poly_actual = sleef_poly_eval_11(x_local, x_power2, x_power4, x_power8, coeffs_ptr); }
            else if constexpr (Degree == 12) { sleef_poly_actual = sleef_poly_eval_12(x_local, x_power2, x_power4, x_power8, coeffs_ptr); }
            else if constexpr (Degree == 13) { sleef_poly_actual = sleef_poly_eval_13(x_local, x_power2, x_power4, x_power8, coeffs_ptr); }
            else if constexpr (Degree == 14) { sleef_poly_actual = sleef_poly_eval_14(x_local, x_power2, x_power4, x_power8, coeffs_ptr); }
            else if constexpr (Degree == 15) { sleef_poly_actual = sleef_poly_eval_15(x_local, x_power2, x_power4, x_power8, coeffs_ptr); }

            check_accuracy("Unrolled Estrin", sleef_poly_actual, expected_result, test_x);
        }
    }
    std::cout << "-------------------------------------------\n";

    all_results.push_back(current_degree_results); // Store results
  }

  // Prints the summary table of all benchmark results.
  static void print_summary_table() {
    std::cout << "\n\n--- Summary Benchmarking Results (ns/eval) ---\n";
    std::cout << std::fixed << std::setprecision(2);

    // Header for the table
    std::cout << std::setw(8) << std::left << "Degree"
        << std::setw(15) << std::right << "Estrin's"
        << std::setw(15) << std::right << "Horner (Ptr)"
        << std::setw(18) << std::right << "Unrolled Estrin"
        << "\n";
    std::cout << std::string(66, '-') << "\n"; // Separator line

    // Iterate through collected results and print them.
    for (const auto &res : all_results) {
      std::cout << std::setw(8) << std::left << res.degree
          << std::setw(15) << std::right << res.estrin_ns_per_eval
          << std::setw(15) << std::right << res.horner_ns_per_eval;

      if (res.sleef_poly_eval_ns_per_eval.has_value()) {
        std::cout << std::setw(18) << std::right << *res.sleef_poly_eval_ns_per_eval;
      } else {
        std::cout << std::setw(18) << std::right << "N/A";
      }
      std::cout << "\n";
    }
    std::cout << std::string(66, '-') << "\n"; // Closing separator line
  }
};

int main() {
  // Use a smaller x_val to ensure numerical stability for higher degrees.
  // Values within [-1, 1] are generally preferred for standard polynomial bases.
  volatile double x_val = 0.5;

  // Set up random number generation for coefficients between -1.0 and 1.0
  std::mt19937_64 rng(std::chrono::steady_clock::now().time_since_epoch().count()); // Seed with current time
  std::uniform_real_distribution<double> dist(-1.0, 1.0); // Distribution for [-1.0, 1.0]

  // Define the maximum number of coefficients needed (for degree 32, we need 33 coefficients)
  constexpr std::size_t MAX_COEFFS = 33;
  std::array<double, MAX_COEFFS> coeffs_array{};

  // Fill coeffs_array with random values between -1.0 and 1.0
  std::generate(coeffs_array.begin(), coeffs_array.end(), [&]() { return dist(rng); });

  const double *coeffs_ptr = coeffs_array.data();

  // Number of iterations for benchmarking. Adjust as needed for desired precision vs. runtime.
  const int N_ITERATIONS = 10000000;

  // Run benchmarks for degrees 0 to 32.
  PolynomialBenchmarkSuite<
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
  >::run_all(x_val, coeffs_ptr, N_ITERATIONS);

  return 0;
}
