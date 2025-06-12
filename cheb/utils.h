#pragma once
#if __cplusplus >= 202002L // Check for C++20 or later
#include <bit>
#endif
#include <cstdint>
#include "macros.h"

namespace poly_eval::detail {
template <typename T>
ALWAYS_INLINE static constexpr size_t get_alignment(const T *ptr) noexcept {
  const auto address = reinterpret_cast<uintptr_t>(ptr);

  if (address == 0) {
    // A null pointer (or an address of 0) doesn't have a meaningful alignment
    // in the context of data access. You could return 0 or 1 depending on
    // your specific interpretation, but 0 makes sense here.
    return 0;
  }
  // Modern C++ (C++20 and later) has built-in functions for this
#if __cplusplus >= 202002L // Check for C++20 or later
  // std::countr_zero returns the number of trailing zero bits.
  // If an address is N-byte aligned, its N lowest bits must be zero.
  // So, if an address is 8-byte aligned (e.g., 0x...1000), it has 3 trailing zeros.
  // 2^3 = 8.
  return static_cast<size_t>(1) << std::countr_zero(address);
#else
  // For older C++ versions, we can use a loop or compiler intrinsics.
  // This loop finds the least significant '1' bit, which determines the alignment.
  size_t alignment = 1;
  while ((address & alignment) == 0 && (alignment < (static_cast<size_t>(1) << (sizeof(uintptr_t) * 8 - 1)))) {
    alignment <<= 1; // Multiply by 2
  }
  return alignment;
#endif
}

template <typename F, std::size_t... Is>
ALWAYS_INLINE constexpr void unroll_loop_impl(F &&func, std::index_sequence<Is...>) {
  (func(Is), ...); // C++17 fold expression for comma operator
}

template <std::size_t Count, typename F>
ALWAYS_INLINE constexpr void unroll_loop(F &&func) {
  unroll_loop_impl(std::forward<F>(func), std::make_index_sequence<Count>{});
}


}