#include "fast_eval.hpp"
#include <array>
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

// Benchmark group<N>() using its batched operator()(x_ptr, out_ptr, count)
template <std::size_t N> static void bench_group_batch(const std::vector<double> &x, ankerl::nanobench::Bench &bench) {
  auto group = make_group<N>();
  constexpr std::size_t kF_pad = N;

  std::vector<double> out(x.size() * kF_pad);

  bench.run(std::to_string(N) + " funcs × " + std::to_string(x.size()) + " pts", [&] {
    group(x.data(), out.data(), x.size());
    ankerl::nanobench::doNotOptimizeAway(out);
  });
}

// Dispatcher for N = 1..MaxFuncs
template <std::size_t MaxFuncs>
static bool dispatch(std::size_t n, const std::vector<double> &x, ankerl::nanobench::Bench &bench) {
  if constexpr (MaxFuncs == 0) {
    return false;
  } else {
    if (n == MaxFuncs) {
      bench_group_batch<MaxFuncs>(x, bench);
      return true;
    }
    return dispatch<MaxFuncs - 1>(n, x, bench);
  }
}

int main() {
  constexpr std::size_t MaxFuncs = 16;
  const std::vector<std::size_t> sizes = {8, 32, 128, 512};

  std::mt19937 rng{42};
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  // Benchmark for each point count
  for (std::size_t n_pts : sizes) {
    // Generate random inputs
    std::vector<double> pts(n_pts);
    for (auto &p : pts)
      p = dist(rng);

    // Benchmark title and setup
    ankerl::nanobench::Bench bench;
    bench.title("poly_eval many-x operator()").unit("eval").warmup(10).minEpochIterations(20'000);

    std::cout << "---- Benchmarking with " << n_pts << " points ----\n";
    for (std::size_t n_funcs = 1; n_funcs <= MaxFuncs; ++n_funcs) {
      bench.batch(n_pts * n_funcs); // Set batch size for this run
      dispatch<MaxFuncs>(n_funcs, pts, bench);
    }
  }

  return 0;
}
