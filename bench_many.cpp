#include "fast_eval.hpp"
#include <nanobench.h>
#include <cmath>
#include <iostream>
#include <random>
#include <numeric> // For std::accumulate
#include <vector>  // For std::vector
#include <xsimd/xsimd.hpp> // For xsimd::aligned_allocator and xsimd::aligned_free (if needed for older xsimd)
#include <iomanip> // For std::setprecision

int main() {
  // Non-constexpr declarations that do not immediately depend on constexprs
  ankerl::nanobench::Bench bench;
  std::mt19937 rng{42};

  // Original 1024 points
  std::array<double, 1024> random_points{};
  std::array<double, 1024> output{};
  std::vector<double> unaligned_random_points(1024);
  std::vector<double> unaligned_output(1024);
  std::vector<double, xsimd::aligned_allocator<double, 64>> aligned_random_points(1024);
  std::vector<double, xsimd::aligned_allocator<double, 64>> aligned_output(1024);

  // New 1000 points case
  constexpr size_t num_points_1000 = 1000;
  std::array<double, num_points_1000> random_points_1000{};
  std::array<double, num_points_1000> output_1000{};
  std::vector<double> unaligned_random_points_1000(num_points_1000);
  std::vector<double> unaligned_output_1000(num_points_1000);
  std::vector<double, xsimd::aligned_allocator<double, 64>> aligned_random_points_1000(num_points_1000);
  std::vector<double, xsimd::aligned_allocator<double, 64>> aligned_output_1000(num_points_1000);


  // Essential constexpr declarations (placed just before their dependent uses)
  constexpr auto f = [](double x) { return std::cos(x); };
  constexpr double a = -.5;
  constexpr double b = .5;
  constexpr auto degree = 32;

  // Now, declare objects that depend on the above constexprs
  const auto evaluator = poly_eval::make_func_eval(f, degree, a, b);
  constexpr auto fixed_evaluator = poly_eval::make_func_eval<degree>(f, a, b);

  // Other non-constexpr declarations that depend on previously defined constexprs
  std::uniform_real_distribution<double> dist{a, b}; // Depends on 'a' and 'b'

  bench.title("Monomial Vectorization Benchmark").unit("evals").warmup(1'000).relative(false).
        minEpochIterations(5'000); // Batch size will be set per run

  // Populate all data sets (1024 points)
  for (auto &pt : random_points) {
    pt = dist(rng);
  }
  for (auto &pt : unaligned_random_points) {
    pt = dist(rng);
  }
  for (auto &pt : aligned_random_points) {
    pt = dist(rng);
  }

  // Populate all data sets (1000 points)
  for (auto &pt : random_points_1000) {
    pt = dist(rng);
  }
  for (auto &pt : unaligned_random_points_1000) {
    pt = dist(rng);
  }
  for (auto &pt : aligned_random_points_1000) {
    pt = dist(rng);
  }

  using ankerl::nanobench::detail::doNotOptimizeAway;

  // --- NON-CONSTEXPR Benchmarks (1024 points) ---
  bench.batch(1024);

  // Benchmarks with std::array (often stack-allocated, alignment can vary)
  bench.run("auto vectorization (std::array, 1024)", [&] {
    for (size_t i = 0; i < random_points.size(); ++i) {
      output[i] = evaluator(random_points[i]);
    }
  });
  const auto sum_auto = std::accumulate(output.begin(), output.end(), 0.0);
  std::ranges::fill(output, 0.0);

  bench.run("manual vectorization (std::array, 1024)", [&] {
    evaluator.operator()<true, true>(random_points.data(), output.data(), random_points.size());
  });
  const auto sum_manual = std::accumulate(output.begin(), output.end(), 0.0);
  std::ranges::fill(output, 0.0);

  // Benchmarks with Unaligned std::vector
  bench.run("auto vectorization (unaligned, 1024)", [&] {
    for (size_t i = 0; i < unaligned_random_points.size(); ++i) {
      unaligned_output[i] = evaluator(unaligned_random_points[i]);
    }
  });
  const auto sum_unaligned_auto = std::accumulate(unaligned_output.begin(), unaligned_output.end(), 0.0);
  std::ranges::fill(unaligned_output, 0.0);

  bench.run("manual vectorization (unaligned, 1024)", [&] {
    evaluator(unaligned_random_points.data(), unaligned_output.data(), unaligned_random_points.size());
  });
  const auto sum_unaligned_manual = std::accumulate(unaligned_output.begin(), unaligned_output.end(), 0.0);
  std::ranges::fill(unaligned_output, 0.0);

  // Benchmarks with Aligned xsimd::aligned_allocator
  bench.run("auto vectorization (aligned, 1024)", [&] {
    for (size_t i = 0; i < aligned_random_points.size(); ++i) {
      aligned_output[i] = evaluator(aligned_random_points[i]);
    }
  });
  const auto sum_aligned_auto = std::accumulate(aligned_output.begin(), aligned_output.end(), 0.0);
  std::ranges::fill(aligned_output, 0.0);

  bench.run("manual vectorization (aligned, 1024)", [&] {
    evaluator.operator()<true, true>(aligned_random_points.data(), aligned_output.data(), aligned_random_points.size());
  });
  const auto sum_aligned_manual = std::accumulate(aligned_output.begin(), aligned_output.end(), 0.0);
  std::ranges::fill(aligned_output, 0.0);


  // --- NON-CONSTEXPR Benchmarks (1000 points) ---
  bench.batch(num_points_1000);

  // Benchmarks with std::array (often stack-allocated, alignment can vary)
  bench.run("auto vectorization (std::array, 1000)", [&] {
    for (size_t i = 0; i < random_points_1000.size(); ++i) {
      output_1000[i] = evaluator(random_points_1000[i]);
    }
  });
  const auto sum_auto_1000 = std::accumulate(output_1000.begin(), output_1000.end(), 0.0);
  std::ranges::fill(output_1000, 0.0);

  bench.run("manual vectorization (std::array, 1000)", [&] {
    evaluator.operator()<true, true>(random_points_1000.data(), output_1000.data(), random_points_1000.size());
  });
  const auto sum_manual_1000 = std::accumulate(output_1000.begin(), output_1000.end(), 0.0);
  std::ranges::fill(output_1000, 0.0);

  // Benchmarks with Unaligned std::vector
  bench.run("auto vectorization (unaligned, 1000)", [&] {
    for (size_t i = 0; i < unaligned_random_points_1000.size(); ++i) {
      unaligned_output_1000[i] = evaluator(unaligned_random_points_1000[i]);
    }
  });
  const auto sum_unaligned_auto_1000 = std::accumulate(unaligned_output_1000.begin(), unaligned_output_1000.end(), 0.0);
  std::ranges::fill(unaligned_output_1000, 0.0);

  bench.run("manual vectorization (unaligned, 1000)", [&] {
    evaluator(unaligned_random_points_1000.data(), unaligned_output_1000.data(), unaligned_random_points_1000.size());
  });
  const auto sum_unaligned_manual_1000 = std::accumulate(unaligned_output_1000.begin(), unaligned_output_1000.end(), 0.0);
  std::ranges::fill(unaligned_output_1000, 0.0);

  // Benchmarks with Aligned xsimd::aligned_allocator
  bench.run("auto vectorization (aligned, 1000)", [&] {
    for (size_t i = 0; i < aligned_random_points_1000.size(); ++i) {
      aligned_output_1000[i] = evaluator(aligned_random_points_1000[i]);
    }
  });
  const auto sum_aligned_auto_1000 = std::accumulate(aligned_output_1000.begin(), aligned_output_1000.end(), 0.0);
  std::ranges::fill(aligned_output_1000, 0.0);

  bench.run("manual vectorization (aligned, 1000)", [&] {
    evaluator.operator()<true, true>(aligned_random_points_1000.data(), aligned_output_1000.data(), aligned_random_points_1000.size());
  });
  const auto sum_aligned_manual_1000 = std::accumulate(aligned_output_1000.begin(), aligned_output_1000.end(), 0.0);
  std::ranges::fill(aligned_output_1000, 0.0);


  // --- CONSTEXPR Benchmarks (1024 points) ---
  bench.batch(1024);

  // Benchmarks with std::array (often stack-allocated, alignment can vary)
  bench.run("constexpr auto vectorization (std::array, 1024)", [&] {
    for (size_t i = 0; i < random_points.size(); ++i) {
      output[i] = fixed_evaluator(random_points[i]);
    }
  });
  const auto sum_auto_const = std::accumulate(output.begin(), output.end(), 0.0);
  std::ranges::fill(output, 0.0);

  bench.run("constexpr manual vectorization (std::array, 1024)", [&] {
    fixed_evaluator(random_points.data(), output.data(), random_points.size());
  });
  const auto sum_manual_const = std::accumulate(output.begin(), output.end(), 0.0);
  std::ranges::fill(output, 0.0);

  // Benchmarks with Unaligned std::vector
  bench.run("constexpr auto vectorization (unaligned, 1024)", [&] {
    for (size_t i = 0; i < unaligned_random_points.size(); ++i) {
      unaligned_output[i] = fixed_evaluator(unaligned_random_points[i]);
    }
  });
  const auto sum_unaligned_auto_const = std::accumulate(unaligned_output.begin(), unaligned_output.end(), 0.0);
  std::ranges::fill(unaligned_output, 0.0);

  bench.run("constexpr manual vectorization (unaligned, 1024)", [&] {
    fixed_evaluator(unaligned_random_points.data(), unaligned_output.data(), unaligned_random_points.size());
  });
  const auto sum_unaligned_manual_const = std::accumulate(unaligned_output.begin(), unaligned_output.end(), 0.0);
  std::ranges::fill(unaligned_output, 0.0);

  // Benchmarks with Aligned xsimd::aligned_allocator
  bench.run("constexpr auto vectorization (aligned, 1024)", [&] {
    for (size_t i = 0; i < aligned_random_points.size(); ++i) {
      aligned_output[i] = fixed_evaluator(aligned_random_points[i]);
    }
  });
  const auto sum_aligned_auto_const = std::accumulate(aligned_output.begin(), aligned_output.end(), 0.0);
  std::ranges::fill(aligned_output, 0.0);

  bench.run("constexpr manual vectorization (aligned, 1024)", [&] {
    fixed_evaluator(aligned_random_points.data(), aligned_output.data(), aligned_random_points.size());
  });
  const auto sum_aligned_manual_const = std::accumulate(aligned_output.begin(), aligned_output.end(), 0.0);
  std::ranges::fill(aligned_output, 0.0);


  // --- CONSTEXPR Benchmarks (1000 points) ---
  bench.batch(num_points_1000);

  // Benchmarks with std::array (often stack-allocated, alignment can vary)
  bench.run("constexpr auto vectorization (std::array, 1000)", [&] {
    for (size_t i = 0; i < random_points_1000.size(); ++i) {
      output_1000[i] = fixed_evaluator(random_points_1000[i]);
    }
  });
  const auto sum_auto_const_1000 = std::accumulate(output_1000.begin(), output_1000.end(), 0.0);
  std::ranges::fill(output_1000, 0.0);

  bench.run("constexpr manual vectorization (std::array, 1000)", [&] {
    fixed_evaluator(random_points_1000.data(), output_1000.data(), random_points_1000.size());
  });
  const auto sum_manual_const_1000 = std::accumulate(output_1000.begin(), output_1000.end(), 0.0);
  std::ranges::fill(output_1000, 0.0);

  // Benchmarks with Unaligned std::vector
  bench.run("constexpr auto vectorization (unaligned, 1000)", [&] {
    for (size_t i = 0; i < unaligned_random_points_1000.size(); ++i) {
      unaligned_output_1000[i] = fixed_evaluator(unaligned_random_points_1000[i]);
    }
  });
  const auto sum_unaligned_auto_const_1000 = std::accumulate(unaligned_output_1000.begin(), unaligned_output_1000.end(), 0.0);
  std::ranges::fill(unaligned_output_1000, 0.0);

  bench.run("constexpr manual vectorization (unaligned, 1000)", [&] {
    fixed_evaluator(unaligned_random_points_1000.data(), unaligned_output_1000.data(), unaligned_random_points_1000.size());
  });
  const auto sum_unaligned_manual_const_1000 = std::accumulate(unaligned_output_1000.begin(), unaligned_output_1000.end(), 0.0);
  std::ranges::fill(unaligned_output_1000, 0.0);

  // Benchmarks with Aligned xsimd::aligned_allocator
  bench.run("constexpr auto vectorization (aligned, 1000)", [&] {
    for (size_t i = 0; i < aligned_random_points_1000.size(); ++i) {
      aligned_output_1000[i] = fixed_evaluator(aligned_random_points_1000[i]);
    }
  });
  const auto sum_aligned_auto_const_1000 = std::accumulate(aligned_output_1000.begin(), aligned_output_1000.end(), 0.0);
  std::ranges::fill(aligned_output_1000, 0.0);

  bench.run("constexpr manual vectorization (aligned, 1000)", [&] {
    fixed_evaluator(aligned_random_points_1000.data(), aligned_output_1000.data(), aligned_random_points_1000.size());
  });
  const auto sum_aligned_manual_const_1000 = std::accumulate(aligned_output_1000.begin(), aligned_output_1000.end(), 0.0);
  std::ranges::fill(aligned_output_1000, 0.0);


  std::cout << std::scientific << std::setprecision(16);
  std::cout << "\n--- Sums from different benchmarks (1024 points) ---\n";
  std::cout << "Sum auto (std::array): " << sum_auto << "\n";
  std::cout << "Sum manual (std::array): " << sum_manual << "\n";
  std::cout << "Sum auto (unaligned): " << sum_unaligned_auto << "\n";
  std::cout << "Sum manual (unaligned): " << sum_unaligned_manual << "\n";
  std::cout << "Sum auto (aligned): " << sum_aligned_auto << "\n";
  std::cout << "Sum manual (aligned): " << sum_aligned_manual << "\n";

  std::cout << "\n--- Sums from different benchmarks (1000 points) ---\n";
  std::cout << "Sum auto (std::array, 1000): " << sum_auto_1000 << "\n";
  std::cout << "Sum manual (std::array, 1000): " << sum_manual_1000 << "\n";
  std::cout << "Sum auto (unaligned, 1000): " << sum_unaligned_auto_1000 << "\n";
  std::cout << "Sum manual (unaligned, 1000): " << sum_unaligned_manual_1000 << "\n";
  std::cout << "Sum auto (aligned, 1000): " << sum_aligned_auto_1000 << "\n";
  std::cout << "Sum manual (aligned, 1000): " << sum_aligned_manual_1000 << "\n";

  std::cout << "\n--- Sums from different benchmarks (1024 points, CONSTEXPR) ---\n";
  std::cout << "Sum auto constexpr (std::array): " << sum_auto_const << "\n";
  std::cout << "Sum manual constexpr (std::array): " << sum_manual_const << "\n";
  std::cout << "Sum auto constexpr (unaligned): " << sum_unaligned_auto_const << "\n";
  std::cout << "Sum manual constexpr (unaligned): " << sum_unaligned_manual_const << "\n";
  std::cout << "Sum auto constexpr (aligned): " << sum_aligned_auto_const << "\n";
  std::cout << "Sum manual constexpr (aligned): " << sum_aligned_manual_const << "\n";

  std::cout << "\n--- Sums from different benchmarks (1000 points, CONSTEXPR) ---\n";
  std::cout << "Sum auto constexpr (std::array, 1000): " << sum_auto_const_1000 << "\n";
  std::cout << "Sum manual constexpr (std::array, 1000): " << sum_manual_const_1000 << "\n";
  std::cout << "Sum auto constexpr (unaligned, 1000): " << sum_unaligned_auto_const_1000 << "\n";
  std::cout << "Sum manual constexpr (unaligned, 1000): " << sum_unaligned_manual_const_1000 << "\n";
  std::cout << "Sum auto constexpr (aligned, 1000): " << sum_aligned_auto_const_1000 << "\n";
  std::cout << "Sum manual constexpr (aligned, 1000): " << sum_aligned_manual_const_1000 << "\n";

  return 0;
}