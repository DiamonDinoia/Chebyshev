#include "fast_eval.hpp"
#include <cmath>
#include <iostream>
#include <nanobench.h>
#include <random>
#include <vector>

// Helper to build a FuncEvalMany with N identical sin evaluators
template <std::size_t N, std::size_t... Is> static auto make_group_impl(std::index_sequence<Is...>) {
  auto make_one = [] { return poly_eval::make_func_eval([](double x) { return std::sin(x); }, 16, -1.0, 1.0); };
  return poly_eval::make_func_eval((static_cast<void>(Is), make_one())...);
}

template <std::size_t N> static auto make_group() { return make_group_impl<N>(std::make_index_sequence<N>{}); }

// Benchmark a group with N functions over the given input points
template <std::size_t N> static void bench_group(const std::vector<double> &pts, ankerl::nanobench::Bench &bench) {
  auto group = make_group<N>();
  bench.run(std::to_string(N) + " funcs", [&] {
    for (double x : pts) {
      auto tup = group(x);
      ankerl::nanobench::doNotOptimizeAway(tup); // Prevent optimization of the result
    }
  });
}

// Compile-time dispatcher up to MaxN
template <std::size_t MaxN>
static bool dispatch(std::size_t n, const std::vector<double> &pts, ankerl::nanobench::Bench &bench) {
  if constexpr (MaxN == 0) {
    return false;
  } else {
    if (n == MaxN) {
      bench_group<MaxN>(pts, bench);
      return true;
    }
    return dispatch<MaxN - 1>(n, pts, bench);
  }
}

int main(int argc, char **argv) {
  /* ---------- 1. generate the inputs once -------------------------------- */
  constexpr std::size_t num_points = 1024;
  std::mt19937 rng{42};
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  std::vector<double> pts(num_points);
  for (auto &p : pts)
    p = dist(rng);

  /* ---------- 2. configure the benchmark object once --------------------- */
  ankerl::nanobench::Bench bench;
  bench.title("poly_eval grouped-function throughput")
      .unit("eval")
      .warmup(100)
      .minEpochIterations(10'000)
      .batch(num_points);

  /* ---------- 3. run N = 1 .. 16 ----------------------------------------- */
  for (std::size_t n = 1; n <= 16; ++n)
    dispatch<16>(n, pts, bench); // always succeeds for 1‒16
}