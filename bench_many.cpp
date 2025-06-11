#include "fast_eval.h"
#include <nanobench.h>
#include <cmath>
#include <iostream>
#include <random>


int main() {
  constexpr auto f = [](double x) { return std::cos(x); };
  constexpr double a = -.5;
  constexpr double b = .5;
  constexpr auto degree = 32;
  const auto evaluator = poly_eval::make_func_eval(f, degree, a, b);
  constexpr auto fixed_evaluator = poly_eval::make_constexpr_func_eval<degree>(f, a, b);
  ankerl::nanobench::Bench bench;
  bench.title("Monomial Vectorization Benchmark").unit("evals").warmup(10000).relative(true).
        minEpochIterations(50'000).batch(1024);
  std::mt19937 rng{42};
  std::uniform_real_distribution<double> dist{a, b};

  std::array<double, 1024> random_points{};
  std::array<double, 1024> output{};
  for (auto &pt : random_points) {
    pt = dist(rng);
  }

  bench.run("auto vectorization", [&] {
    for (size_t i = 0; i < random_points.size(); ++i) {
      output[i] = evaluator(random_points[i]);
    }
  });
  bench.run("manual vectorization", [&] {
    evaluator(random_points.data(), output.data(), random_points.size());
  });
  bench.run("constexpr auto vectorization", [&] {
    for (size_t i = 0; i < random_points.size(); ++i) {
      output[i] = fixed_evaluator(random_points[i]);
    }
  });
  bench.run("constexpr manual vectorization", [&] {
    fixed_evaluator(random_points.data(), output.data(), random_points.size());
  });

  return 0;
}