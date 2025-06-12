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
