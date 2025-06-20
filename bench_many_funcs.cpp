// bench_poly_eval.cppIs
//
// Benchmark poly_eval::make_func_eval groups with ankerl::nanobench
//
// g++ -std=c++20 -O3 -march=native -I/path/to/nanobench -I/path/to/poly_eval \
//     bench_poly_eval.cpp -o bench_poly_eval
//
// Run: ./bench_poly_eval --json benchmark.json   (optional nanobench CLI flags)

#include <nanobench.h>          // ankerl::nanobench
#include "fast_eval.hpp"        // your poly_eval utilities
#include <cmath>                // std::sin / std::cos
#include <array>
#include <tuple>
#include <string_view>

// -----------------------------------------------------------------------------
// 1.  Helper that makes N identical sin-evaluators and groups them.
//     The pack-expansion trick avoids writing N by hand.
// -----------------------------------------------------------------------------
template <std::size_t N, std::size_t... Is>
static auto make_group_impl(std::index_sequence<Is...>) {
    // Create one evaluator inside the expansion — each copy is independent.
    auto make_one = [] {
        return poly_eval::make_func_eval(
            [](double x) { return std::sin(x); }, 16 /*deg*/, -1.0, 1.0);
    };
    return poly_eval::make_func_eval( (static_cast<void>(Is), make_one())... );
}

template <std::size_t N>
static auto make_group() {
    return make_group_impl<N>(std::make_index_sequence<N>{});
}

// -----------------------------------------------------------------------------
// 2.  Single-benchmark helper
// -----------------------------------------------------------------------------
template <std::size_t N>
static void bench_group(ankerl::nanobench::Bench& bench) {
    auto group = make_group<N>();
    double x = 0.42;
    bench.run(std::to_string(N) + " funcs", [&] {
        // Evaluate all N funcs on the *same* x; change if you need different x.
      group(x);
    });
}

// -----------------------------------------------------------------------------
// 3.  Drive benchmarks for any constexpr list of sizes
// -----------------------------------------------------------------------------
int main() {
    ankerl::nanobench::Bench bench;
    bench.title("poly_eval grouped-function throughput");
    bench.unit("eval");          // each run counts 1 group evaluation
    bench.warmup(100);
    bench.minEpochIterations(10'000);

    // <<< Edit this list or turn it into a constexpr array to sweep sizes >>>
    bench_group<2>(bench);
    bench_group<4>(bench);
    bench_group<8>(bench);
    bench_group<16>(bench);
}
