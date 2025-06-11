#include <iostream>
#include <vector>
#include <cmath>     // For std::fabs, std::pow, std::fma
#include <array>     // For std::array for coefficients
#include <chrono>    // For benchmarking
#include <string>    // For std::string in Benchmark
#include <tuple>     // For std::make_tuple, std::get
#include <utility>   // For std::index_sequence, std::make_index_sequence
#include <optional>  // For std::optional
#include <iomanip>   // For std::setw, std::left, std::right, std::fixed, std::setprecision

// --- Common Helper Function ---
template <typename T>
__always_inline constexpr T fma(T a, T b, T c) {
  return std::fma(a, b, c);
}

// --- Compile-time Log2 function ---
template <std::size_t N>
__always_inline constexpr std::size_t ct_log2() {
  static_assert(N > 0, "ct_log2 of zero is undefined.");
  if constexpr (N == 1) {
    return 0; // log2(1) = 0
  } else {
    return ct_log2<(N / 2)>() + 1;
  }
}

// --- Helper to compute x^(2^k) at compile time for building the initial powers tuple ---
template <typename T, std::size_t K>
__always_inline constexpr T compute_x_power_of_two(T x) {
  if constexpr (K == 0) {
    return x; // x^(2^0) = x^1
  } else {
    T prev_power = compute_x_power_of_two<T, K - 1>(x);
    return prev_power * prev_power; // x^(2^K) = (x^(2^(K-1)))^2
  }
}

// --- Helper to create tuple of powers for Estrin's scheme ---
template <typename T, std::size_t... K>
__always_inline constexpr auto make_powers_tuple(T val, std::index_sequence<K...>) {
  return std::make_tuple(compute_x_power_of_two<T, K>(val)...);
}

// --- Estrin's Scheme Recursive Evaluation ---
template <typename T, typename PowersTuple, std::size_t Count>
__always_inline constexpr T estrin_eval_recursive(
    T x, PowersTuple powers_tuple, const T *coeffs) {
  if constexpr (Count == 0) {
    return T{};
  } else if constexpr (Count == 1) {
    return coeffs[0]; // Degree 0: P(x) = c0
  } else {
    // Find the largest power of 2 (P2) such that P2 <= (Count - 1)
    constexpr std::size_t current_degree = Count - 1;
    constexpr std::size_t split_power_of_2 =
        (current_degree == 0) ? 1 : (1ULL << ct_log2<current_degree>());

    constexpr std::size_t x_power_log2_idx = ct_log2<split_power_of_2>();

    // P(x) = P_lower(x) + x^P2 * P_higher(x)
    return fma(std::get<x_power_log2_idx>(powers_tuple),
               estrin_eval_recursive<T, PowersTuple, Count - split_power_of_2>(
                   x, powers_tuple, coeffs + split_power_of_2),
               estrin_eval_recursive<T, PowersTuple, split_power_of_2>(x, powers_tuple, coeffs));
  }
}

// --- Public Interface for Estrin's Polynomial Evaluation ---
template <typename T, std::size_t N_coeffs>
__always_inline constexpr T estrin_eval(T x, const T *coeffs) {
  static_assert(N_coeffs > 0, "At least one coefficient (c0) is required for a polynomial.");

  if constexpr (N_coeffs == 1) {
    return coeffs[0]; // Special case for degree 0
  } else {
    constexpr std::size_t max_degree = N_coeffs - 1;
    constexpr std::size_t max_k_value_for_powers_tuple = ct_log2<max_degree>();

    const auto powers_tuple = make_powers_tuple(x, std::make_index_sequence<max_k_value_for_powers_tuple + 1>{});

    return estrin_eval_recursive<T, decltype(powers_tuple), N_coeffs>(x, powers_tuple, coeffs);
  }
}

// --- Unrolled Estrin's Macro-like Functions (for degrees 0-15) ---
// These mimic unrolled Estrin implementations often found in specific libraries like SLEEF.
template <typename T>
__always_inline constexpr T sleef_poly_eval_0(T /*x*/, const T *coeffs) { // N_coeffs = 1
    return coeffs[0];
}

template <typename T>
__always_inline constexpr T sleef_poly_eval_1(T x, const T *coeffs) { // N_coeffs = 2
    return fma(x, coeffs[1], coeffs[0]);
}

template <typename T>
__always_inline constexpr T sleef_poly_eval_2(T x, T x2, const T *coeffs) { // N_coeffs = 3
    return fma(x2, coeffs[2], fma(x, coeffs[1], coeffs[0]));
}

template <typename T>
__always_inline constexpr T sleef_poly_eval_3(T x, T x2, const T *coeffs) { // N_coeffs = 4
    return fma(x2, fma(x, coeffs[3], coeffs[2]), fma(x, coeffs[1], coeffs[0]));
}

template <typename T>
__always_inline constexpr T sleef_poly_eval_4(T x, T x2, T x4, const T *coeffs) { // N_coeffs = 5
    return fma(x4, coeffs[4], sleef_poly_eval_3(x, x2, coeffs));
}

template <typename T>
__always_inline constexpr T sleef_poly_eval_5(T x, T x2, T x4, const T *coeffs) { // N_coeffs = 6
    return fma(x4, sleef_poly_eval_1(x, coeffs + 4), sleef_poly_eval_3(x, x2, coeffs));
}

template <typename T>
__always_inline constexpr T sleef_poly_eval_6(T x, T x2, T x4, const T *coeffs) { // N_coeffs = 7
    return fma(x4, sleef_poly_eval_2(x, x2, coeffs + 4), sleef_poly_eval_3(x, x2, coeffs));
}

template <typename T>
__always_inline constexpr T sleef_poly_eval_7(T x, T x2, T x4, const T *coeffs) { // N_coeffs = 8
    return fma(x4, sleef_poly_eval_3(x, x2, coeffs + 4), sleef_poly_eval_3(x, x2, coeffs));
}

template <typename T>
__always_inline constexpr T sleef_poly_eval_8(T x, T x2, T x4, T x8, const T *coeffs) { // N_coeffs = 9
    return fma(x8, coeffs[8], sleef_poly_eval_7(x, x2, x4, coeffs));
}

template <typename T>
__always_inline constexpr T sleef_poly_eval_9(T x, T x2, T x4, T x8, const T *coeffs) { // N_coeffs = 10
    return fma(x8, sleef_poly_eval_1(x, coeffs + 8), sleef_poly_eval_7(x, x2, x4, coeffs));
}

template <typename T>
__always_inline constexpr T sleef_poly_eval_10(T x, T x2, T x4, T x8, const T *coeffs) { // N_coeffs = 11
    return fma(x8, sleef_poly_eval_2(x, x2, coeffs + 8), sleef_poly_eval_7(x, x2, x4, coeffs));
}

template <typename T>
__always_inline constexpr T sleef_poly_eval_11(T x, T x2, T x4, T x8, const T *coeffs) { // N_coeffs = 12
    return fma(x8, sleef_poly_eval_3(x, x2, coeffs + 8), sleef_poly_eval_7(x, x2, x4, coeffs));
}

template <typename T>
__always_inline constexpr T sleef_poly_eval_12(T x, T x2, T x4, T x8, const T *coeffs) { // N_coeffs = 13
    return fma(x8, sleef_poly_eval_4(x, x2, x4, coeffs + 8), sleef_poly_eval_7(x, x2, x4, coeffs));
}

template <typename T>
__always_inline constexpr T sleef_poly_eval_13(T x, T x2, T x4, T x8, const T *coeffs) { // N_coeffs = 14
    return fma(x8, sleef_poly_eval_5(x, x2, x4, coeffs + 8), sleef_poly_eval_7(x, x2, x4, coeffs));
}

template <typename T>
__always_inline constexpr T sleef_poly_eval_14(T x, T x2, T x4, T x8, const T *coeffs) { // N_coeffs = 15
    return fma(x8, sleef_poly_eval_6(x, x2, x4, coeffs + 8), sleef_poly_eval_7(x, x2, x4, coeffs));
}

template <typename T>
__always_inline constexpr T sleef_poly_eval_15(T x, T x2, T x4, T x8, const T *coeffs) { // N_coeffs = 16
    return fma(x8, sleef_poly_eval_7(x, x2, x4, coeffs + 8), sleef_poly_eval_7(x, x2, x4, coeffs));
}

// --- General Purpose Horner's Method (pointer and compile-time size version) ---
template <typename T, std::size_t N_coeffs>
__always_inline constexpr T horner_eval(T x, const T *coeffs) {
  static_assert(N_coeffs > 0, "At least one coefficient is required for a polynomial.");
  T result = coeffs[N_coeffs - 1];
  if constexpr (N_coeffs > 1) {
    for (std::size_t i = N_coeffs - 1; i-- > 0;) {
      result = fma(x, result, coeffs[i]);
    }
  }
  return result;
}

struct BenchmarkTimer {
  std::string name;
  std::chrono::high_resolution_clock::time_point start_time;

  __attribute_noinline__ BenchmarkTimer(const std::string &n) : name(n) {
    start_time = std::chrono::high_resolution_clock::now();
  }

  __attribute_noinline__ long long stop() {
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
  }
};

struct DegreeBenchmarkResults {
  std::size_t degree;
  double estrin_ns_per_eval;
  double horner_ns_per_eval;
  std::optional<double> sleef_poly_eval_ns_per_eval; // Renamed
};

template <std::size_t... Degrees>
struct PolynomialBenchmarkSuite {
  static inline std::vector<DegreeBenchmarkResults> all_results;

  static void run_all(double x_val, const double *coeffs_ptr, int num_iterations) {
    std::cout << "\n--- Benchmarking Polynomials (Degree 0 to 21) ---\n";
    std::cout << "x = " << x_val << ", Iterations per benchmark = " << num_iterations << "\n\n";

    ((run_single_degree_benchmark<Degrees>(x_val, coeffs_ptr, num_iterations)), ...);

    print_summary_table();
  }

private:
  template <std::size_t Degree>
  static void run_single_degree_benchmark(double x_val, const double *coeffs_ptr, int num_iterations) {
    constexpr std::size_t num_coeffs = Degree + 1;
    DegreeBenchmarkResults current_degree_results;
    current_degree_results.degree = Degree;

    std::cout << "--- Degree " << Degree << " (Coeffs: " << num_coeffs << ") ---\n";

    // --- Estrin's Benchmark ---
    volatile double sum_estrin = 0.0;
    long long estrin_total_ns;
    {
      BenchmarkTimer timer("Estrin's (Deg " + std::to_string(Degree) + ")");
      for (int i = 0; i < num_iterations; ++i) {
        volatile double x = x_val;
        sum_estrin += estrin_eval<double, num_coeffs>(x, coeffs_ptr);
      }
      estrin_total_ns = timer.stop();
      current_degree_results.estrin_ns_per_eval = static_cast<double>(estrin_total_ns) / num_iterations;
    }
    std::cout << "  Estrin Sum: " << sum_estrin << "\n";
    std::cout << "  Estrin's: " << std::fixed << std::setprecision(2) << current_degree_results.estrin_ns_per_eval <<
        " ns/eval\n";

    // --- Horner's (Ptr) Benchmark ---
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
    std::cout << "  Horner (Ptr) Sum: " << sum_horner << "\n";
    std::cout << "  Horner's (Ptr): " << std::fixed << std::setprecision(2) << current_degree_results.
        horner_ns_per_eval << " ns/eval\n";

    // --- Unrolled Estrin Macro Functions Benchmark (using if constexpr) ---
    volatile double sum_sleef_poly = 0.0; // Renamed
    bool sleef_poly_applicable = true;    // Renamed
    long long sleef_poly_total_ns = 0;    // Renamed
    {
      BenchmarkTimer timer("Unrolled Estrin (Deg " + std::to_string(Degree) + ")");
      for (int i = 0; i < num_iterations; ++i) {
        // These must be volatile and computed inside the loop
        // to accurately reflect the overhead of computing powers
        volatile double x_local = x_val;
        volatile double x_power2 = x_local * x_local;
        volatile double x_power4 = x_power2 * x_power2;
        volatile double x_power8 = x_power4 * x_power4;

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
          sleef_poly_applicable = false;
        }
      }
      sleef_poly_total_ns = timer.stop();
      if (sleef_poly_applicable) {
        current_degree_results.sleef_poly_eval_ns_per_eval = static_cast<double>(sleef_poly_total_ns) / num_iterations;
        std::cout << "  Unrolled Estrin Sum: " << sum_sleef_poly << "\n";
        std::cout << "  Unrolled Estrin: " << std::fixed << std::setprecision(2) << *current_degree_results.
            sleef_poly_eval_ns_per_eval << " ns/eval\n";
      } else {
        current_degree_results.sleef_poly_eval_ns_per_eval = std::nullopt;
        std::cout << "  Unrolled Estrin Sum: N/A (No specific unrolled macro for degree " << Degree << ")\n";
        std::cout << "  Unrolled Estrin: N/A\n";
      }
    }

    std::cout << "\n"; // Add extra newline for spacing

    // --- Verification for the current degree ---
    double expected_result = 0.0;
    for (std::size_t i = 0; i < num_coeffs; ++i) {
      expected_result += coeffs_ptr[i] * std::pow(x_val, i);
    }

    std::cout << "  Expected Result (Deg " << Degree << "): " << expected_result << "\n";
    auto check_ratio = [&](const std::string &name_suffix, double val, double expected) {
      if (std::abs(1 - val / expected) < 1e-15) {
        std::cout << "    OK: " << name_suffix << " is very close.\n";
      } else {
        std::cout << "    FAILED: " << name_suffix << " mismatch.\n";
        std::cout << "      Ratio diff: |1 - (Val / Expected)| = " << std::abs(1 - val / expected) << "\n";
        std::cout << "      Absolute diff: " << std::abs(val - expected) << "\n";
      }
    };

    check_ratio("Estrin's", sum_estrin / num_iterations, expected_result);
    check_ratio("Horner's (Ptr)", sum_horner / num_iterations, expected_result);
    if (current_degree_results.sleef_poly_eval_ns_per_eval.has_value()) {
      check_ratio("Unrolled Estrin", sum_sleef_poly / num_iterations, expected_result);
    }
    std::cout << "-------------------------------------------\n";

    all_results.push_back(current_degree_results); // Store results
  }

  static void print_summary_table() {
    std::cout << "\n\n--- Summary Benchmarking Results (ns/eval) ---\n";
    std::cout << std::fixed << std::setprecision(2);

    // Header
    std::cout << std::setw(8) << std::left << "Degree"
        << std::setw(15) << std::right << "Estrin's"
        << std::setw(15) << std::right << "Horner (Ptr)"
        << std::setw(18) << std::right << "Unrolled Estrin"
        << "\n";
    std::cout << std::string(66, '-') << "\n"; // Adjusted line length

    // Data rows
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
    std::cout << std::string(66, '-') << "\n"; // Adjusted line length
  }
};

int main() {
  volatile double x_val = 2.0;

  // Coefficients in ascending order (c0 to c21, for up to degree 21)
  const std::array<double, 22> coeffs_array = {
      1.0, 2.0, 3.0, 4.0, // Deg 0-3
      5.0, 6.0, 7.0, 8.0, // Deg 4-7
      9.0, 10.0, 11.0, 12.0, // Deg 8-11
      13.0, 14.0, 15.0, 16.0, // Deg 12-15
      17.0, 18.0, 19.0, 20.0, // Deg 16-19
      21.0, 22.0 // Deg 20-21
  };

  // Ensure coeffs_array is large enough for degree 21 (22 coeffs)
  static_assert(coeffs_array.size() >= 22, "coeffs_array must have at least 22 elements for degree 21.");

  const double *coeffs_ptr = coeffs_array.data();

  // Adjusted number of iterations for a more reasonable runtime during testing.
  // Feel free to increase this for more precise benchmarks on a high-performance system.
  const int N_ITERATIONS = 10000000;

  PolynomialBenchmarkSuite<
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21
  >::run_all(x_val, coeffs_ptr, N_ITERATIONS);

  return 0;
}