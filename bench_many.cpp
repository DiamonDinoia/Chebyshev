#include "fast_eval.hpp"

#include <nanobench.h>
#include <cmath>
#include <iostream>
#include <random>
#include <numeric> // For std::accumulate
#include <vector>  // For std::vector
#include <xsimd/xsimd.hpp> // For xsimd::aligned_allocator
#include <iomanip> // For std::setprecision

int main() {
  // Non-constexpr declarations that do not immediately depend on constexprs
  ankerl::nanobench::Bench bench;
  std::mt19937 rng{42};

  // Control flags for benchmark execution
  constexpr bool run_non_constexpr_benchmarks = false;
  constexpr bool run_constexpr_benchmarks = !run_non_constexpr_benchmarks;

  // Define the number of points to benchmark
  constexpr size_t num_points = 10000;

  // Data structures for the chosen number of points
  alignas(64) std::array<double, num_points> random_points{};
  alignas(64) std::array<double, num_points> output{};
  std::vector<double> unaligned_random_points(num_points);
  std::vector<double> unaligned_output(num_points);
  std::vector<double, xsimd::aligned_allocator<double, 64>> aligned_random_points(num_points);
  std::vector<double, xsimd::aligned_allocator<double, 64>> aligned_output(num_points);

  // Essential constexpr declarations (placed just before their dependent uses)
  constexpr auto f = [](double x) { return std::cos(x); };
  constexpr double a = -.1;
  constexpr double b = .1;
  constexpr auto degree = 16;

  // Now, declare objects that depend on the above constexprs
  const auto evaluator = poly_eval::make_func_eval(f, degree, a, b);
  constexpr auto fixed_evaluator = poly_eval::make_func_eval<degree>(f, a, b);

  // Other non-constexpr declarations that depend on previously defined constexprs
  std::uniform_real_distribution<double> dist{a, b}; // Depends on 'a' and 'b'

  bench.title("Monomial Vectorization Benchmark").unit("evals").warmup(1'000).relative(false).
        minEpochIterations(5'000).batch(num_points); // Set batch size based on num_points

  // Populate all data sets
  for (auto &pt : random_points) {
    pt = dist(rng);
  }
  std::ranges::copy(random_points, unaligned_random_points.begin());
  std::ranges::copy(random_points, aligned_random_points.begin());

  // --- NON-CONSTEXPR Benchmarks ---
  if (run_non_constexpr_benchmarks) {
    std::cout << "\n--- Running NON-CONSTEXPR Benchmarks (" << num_points << " points) ---\n";

    // Benchmarks with std::array (often stack-allocated, alignment can vary)
    bench.run("auto vectorization (std::array)", [&] {
      for (size_t i = 0; i < random_points.size(); ++i) {
        output[i] = evaluator(random_points[i]);
      }
    });
    const auto sum_auto = std::accumulate(output.begin(), output.end(), 0.0);
    std::ranges::fill(output, 0.0);

    bench.run("manual vectorization (std::array)", [&] {
      evaluator.operator()<true, true>(random_points.data(), output.data(), random_points.size());
    });
    const auto sum_manual = std::accumulate(output.begin(), output.end(), 0.0);
    std::ranges::fill(output, 0.0);

    // Benchmarks with Unaligned std::vector
    bench.run("auto vectorization (unaligned)", [&] {
      for (size_t i = 0; i < unaligned_random_points.size(); ++i) {
        unaligned_output[i] = evaluator(unaligned_random_points[i]);
      }
    });
    const auto sum_unaligned_auto = std::accumulate(unaligned_output.begin(), unaligned_output.end(), 0.0);
    std::ranges::fill(unaligned_output, 0.0);

    bench.run("manual vectorization (unaligned)", [&] {
      evaluator(unaligned_random_points.data(), unaligned_output.data(), unaligned_random_points.size());
    });
    const auto sum_unaligned_manual = std::accumulate(unaligned_output.begin(), unaligned_output.end(), 0.0);
    std::ranges::fill(unaligned_output, 0.0);

    // Benchmarks with Aligned xsimd::aligned_allocator
    bench.run("auto vectorization (aligned)", [&] {
      for (size_t i = 0; i < aligned_random_points.size(); ++i) {
        aligned_output[i] = evaluator(aligned_random_points[i]);
      }
    });
    const auto sum_aligned_auto = std::accumulate(aligned_output.begin(), aligned_output.end(), 0.0);
    std::ranges::fill(aligned_output, 0.0);

    bench.run("manual vectorization (aligned)", [&] {
      evaluator(aligned_random_points.data(), aligned_output.data(),
                                       aligned_random_points.size());
    });
    const auto sum_aligned_manual = std::accumulate(aligned_output.begin(), aligned_output.end(), 0.0);
    std::ranges::fill(aligned_output, 0.0);

    std::cout << std::scientific << std::setprecision(16);
    std::cout << "\n--- Sums from NON-CONSTEXPR benchmarks (" << num_points << " points) ---\n";
    std::cout << "Sum auto (std::array): " << sum_auto << "\n";
    std::cout << "Sum manual (std::array): " << sum_manual << "\n";
    std::cout << "Sum auto (unaligned): " << sum_unaligned_auto << "\n";
    std::cout << "Sum manual (unaligned): " << sum_unaligned_manual << "\n";
    std::cout << "Sum auto (aligned): " << sum_aligned_auto << "\n";
    std::cout << "Sum manual (aligned): " << sum_aligned_manual << "\n";
  }

  // --- CONSTEXPR Benchmarks ---
  if (run_constexpr_benchmarks) {
    std::cout << "\n--- Running CONSTEXPR Benchmarks (" << num_points << " points) ---\n";

    // Benchmarks with std::array (often stack-allocated, alignment can vary)
    bench.run("constexpr auto vectorization (std::array)", [&] {
      for (size_t i = 0; i < random_points.size(); ++i) {
        output[i] = fixed_evaluator(random_points[i]);
      }
    });
    const auto sum_auto_const = std::accumulate(output.begin(), output.end(), 0.0);
    std::ranges::fill(output, 0.0);

    bench.run("constexpr manual vectorization (std::array)", [&] {
      fixed_evaluator(random_points.data(), output.data(), random_points.size());
    });
    const auto sum_manual_const = std::accumulate(output.begin(), output.end(), 0.0);
    std::ranges::fill(output, 0.0);

    // Benchmarks with Unaligned std::vector
    bench.run("constexpr auto vectorization (unaligned)", [&] {
      for (size_t i = 0; i < unaligned_random_points.size(); ++i) {
        unaligned_output[i] = fixed_evaluator(unaligned_random_points[i]);
      }
    });
    const auto sum_unaligned_auto_const = std::accumulate(unaligned_output.begin(), unaligned_output.end(), 0.0);
    std::ranges::fill(unaligned_output, 0.0);

    bench.run("constexpr manual vectorization (unaligned)", [&] {
      fixed_evaluator(unaligned_random_points.data(), unaligned_output.data(), unaligned_random_points.size());
    });
    const auto sum_unaligned_manual_const = std::accumulate(unaligned_output.begin(), unaligned_output.end(), 0.0);
    std::ranges::fill(unaligned_output, 0.0);

    // Benchmarks with Aligned xsimd::aligned_allocator
    bench.run("constexpr auto vectorization (aligned)", [&] {
      for (size_t i = 0; i < aligned_random_points.size(); ++i) {
        aligned_output[i] = fixed_evaluator(aligned_random_points[i]);
      }
    });
    const auto sum_aligned_auto_const = std::accumulate(aligned_output.begin(), aligned_output.end(), 0.0);
    std::ranges::fill(aligned_output, 0.0);

    bench.run("constexpr manual vectorization (aligned)", [&] {
      fixed_evaluator(aligned_random_points.data(), aligned_output.data(), aligned_random_points.size());
    });
    const auto sum_aligned_manual_const = std::accumulate(aligned_output.begin(), aligned_output.end(), 0.0);
    std::ranges::fill(aligned_output, 0.0);

    std::cout << std::scientific << std::setprecision(16);
    std::cout << "\n--- Sums from CONSTEXPR benchmarks (" << num_points << " points) ---\n";
    std::cout << "Sum auto constexpr (std::array): " << sum_auto_const << "\n";
    std::cout << "Sum manual constexpr (std::array): " << sum_manual_const << "\n";
    std::cout << "Sum auto constexpr (unaligned): " << sum_unaligned_auto_const << "\n";
    std::cout << "Sum manual constexpr (unaligned): " << sum_unaligned_manual_const << "\n";
    std::cout << "Sum auto constexpr (aligned): " << sum_aligned_auto_const << "\n";
    std::cout << "Sum manual constexpr (aligned): " << sum_aligned_manual_const << "\n";
  }

  return 0;
}