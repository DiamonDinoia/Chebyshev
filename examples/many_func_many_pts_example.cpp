#include "fast_eval.hpp"
#include <vector>

int main() {
  // Create individual evaluators
  const auto sin = poly_eval::make_func_eval([](double x) { return std::sin(x); }, 16, -1.0, 1.0);
  const auto cos = poly_eval::make_func_eval([](double x) { return std::cos(x); }, 16, -1.0, 1.0);

  // Group them via the maker function
  const auto group = poly_eval::make_func_eval(sin, cos);
  constexpr std::size_t kF_pad = 2;

  // Example batch of inputs
  constexpr std::size_t num_points = 10;
  std::vector<double> inputs(num_points);
  for (std::size_t i = 0; i < num_points; ++i) {
    inputs[i] = -1.0 + 2.0 * i / (num_points - 1);  // linspace
  }

  // Allocate output buffer: [num_points * kF_pad]
  std::vector<double> output(num_points * kF_pad);

  // Evaluate using the batched operator()
  group(inputs.data(), output.data(), num_points);

  // Print results
  for (std::size_t i = 0; i < num_points; ++i) {
    std::cout << "x = " << inputs[i] << " → "
              << "sin = " << output[i * kF_pad + 0]
              << ", cos = " << output[i * kF_pad + 1] << '\n';
    std::cout << "Expected: sin = " << std::sin(inputs[i])
              << ", cos = " << std::cos(inputs[i]) << "\n\n";
  }

  return 0;
}
