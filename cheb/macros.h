#pragma once

// Define a macro for compiler-specific optimization attributes/pragmas
#if defined(__GNUC__) || defined(__clang__)
    // GCC and Clang: Use __attribute__((optimize("-ffast-math")))
    // This single flag enables -fassociative-math, -fno-signed-zeros, -fno-trapping-math,
    // and other aggressive floating-point optimizations.
    #define FAST_MATH_BEGIN \
    __attribute__((optimize("-ffast-math")))
    #define FAST_MATH_END
    // If you wanted to apply it to an entire block or file, you could use:
    // #pragma GCC push_options
    // #pragma GCC optimize("-ffast-math")
    // #pragma GCC pop_options
#elif defined(_MSC_VER)
    // MSVC: Closest general equivalent to -ffast-math via pragmas.
    // #pragma float_control(strict, off) relaxes strict IEEE 754 rules.
    // #pragma fenv_access(off) indicates no reliance on floating-point environment.
    // For signed zeros, MSVC's /fp:fast (compiler option) often treats them as equivalent.
    // There isn't a direct pragma for signed zeros.
    #define FAST_MATH_BEGIN \
    __pragma(float_control(strict, off)) \
    __pragma(fenv_access(off))
    #define FAST_MATH_END \
    __pragma(float_control(strict, on)) \
    __pragma(fenv_access(on))
#elif defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
    // Intel C++ Compiler (ICC/ICX): Has a direct 'fast' float control pragma.
    #define FAST_MATH_BEGIN \
    _Pragma("float_control(fast, on)")
    #define FAST_MATH_END \
    _Pragma("float_control(fast, off)")
#else
    // Fallback for other compilers: no specific fast math pragmas
    #warning "Compiler not recognized, fast math pragmas will not be applied."
    #define FAST_MATH_BEGIN
    #define FAST_MATH_END
#endif


// --- ALWAYS INLINE MACRO ---
#if defined(__GNUC__) || defined(__clang__)
    // GCC and Clang support __attribute__((always_inline))
    #define ALWAYS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
    // MSVC supports __forceinline
    #define ALWAYS_INLINE __forceinline
#else
    // Fallback for other compilers: just use inline.
    // This is a weaker hint, but the best we can do generically.
    #define ALWAYS_INLINE inline
#endif

// --- NO INLINE MACRO ---
#if defined(__GNUC__) || defined(__clang__)
    // GCC and Clang support __attribute__((noinline))
    #define NO_INLINE __attribute__((noinline))
#elif defined(_MSC_VER)
    // MSVC supports __declspec(noinline)
    #define NO_INLINE __declspec(noinline)
#else
    // Fallback for other compilers: no specific attribute.
    // The compiler will decide whether to inline based on its heuristics.
    // This essentially means "don't force inlining or disallow it explicitly".
    #define NO_INLINE
#endif


// Define RESTRICT based on the compiler
#if defined(__GNUC__) || defined(__clang__)
    // GCC and Clang compilers support __restrict__
    #define RESTRICT __restrict__
#elif defined(_MSC_VER)
    // Microsoft Visual C++ compiler supports __restrict
    #define RESTRICT __restrict
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)
    // C99 standard introduced 'restrict' keyword
    #define RESTRICT restrict
#else
    // Fallback for other compilers or older standards
    // In this case, RESTRICT expands to nothing, effectively disabling the keyword.
    // This ensures compilation but without the optimization benefits of restrict.
    #define RESTRICT
#endif

// Define a c++20 cconstexpr macro
#if __cplusplus >= 202002L
    #define C20CONSTEXPR constexpr
#else
#define C20CONSTEXPR
#endif
