#include "cheb1d.h"
#include <nanobench.h>
#include <cmath>
#include <random>

template <typename T, typename V>
void bench_interpolation(ankerl::nanobench::Bench &bench, const std::string &name, const size_t n, V &&f) {

  double a = -1.5, b = 1.5;
  T interpolator(f, n, a, b);

  std::mt19937 rng{42};
  std::uniform_real_distribution<double> dist{a, b};

  bench.run("N=" + std::to_string(n) + " " + name, [&] {
    ankerl::nanobench::doNotOptimizeAway(interpolator(dist(rng)));
  }).minEpochIterations(4000000);

}

int main() {
  auto f = [](double x) { return std::exp(x) + 1; };

  ankerl::nanobench::Bench bench;
  bench.title("Chebyshev Interpolation Benchmark").unit("evals").warmup(100).relative(true);

  for (size_t n = 2; n <= 64; n *= 2) {
    // bench_interpolation<Cheb1D<decltype(f)>>(bench, "Cheb1D", n, f);
    // bench_interpolation<BarCheb1D<decltype(f)>>(bench, "BarCheb1D", n, f);
    bench_interpolation<Hor1D<decltype(f)>>(bench, "Hor1D", n, f);
    bench_interpolation<FixedHor<decltype(f)>>(bench, "FixedHor", n, f);
    bench_interpolation<Est1D<decltype(f)>>(bench, "Est1D", n, f);
    bench_interpolation<FixedEst<decltype(f)>>(bench, "FixedEst", n, f);

  }

  return 0;
}