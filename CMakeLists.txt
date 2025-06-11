cmake_minimum_required(VERSION 3.15)
project(ChebyshevND LANGUAGES CXX)


# ─────────────────────────────────────────────
# Use CPM to fetch nda

set(CPM_DOWNLOAD_VERSION 0.42.0)
set(CPM_HASH_SUM "2020b4fc42dba44817983e06342e682ecfc3d2f484a581f11cc5731fbe4dce8a")

if(CPM_SOURCE_CACHE)
    set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
    set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
    set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

# Expand relative path. This is important if the provided path contains a tilde (~)
get_filename_component(CPM_DOWNLOAD_LOCATION ${CPM_DOWNLOAD_LOCATION} ABSOLUTE)

file(DOWNLOAD
        https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
        ${CPM_DOWNLOAD_LOCATION} EXPECTED_HASH SHA256=${CPM_HASH_SUM}
)

include(${CPM_DOWNLOAD_LOCATION})

CPMAddPackage(
        NAME nda
        GITHUB_REPOSITORY DiamonDinoia/nda
        GIT_TAG refactor-cmake
)

CPMAddPackage(
        NAME xsimd
        GITHUB_REPOSITORY xtensor-stack/xsimd
        GIT_TAG 13.2.0
)

CPMAddPackage(
        nanobench
        GITHUB_REPOSITORY martinus/nanobench
        GIT_TAG v4.3.11
)


add_library(cheb INTERFACE)
target_include_directories(cheb INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cheb)

# ─────────────────────────────────────────────
# Build executable
add_executable(chebyshev_nd main.cpp)

target_link_libraries(chebyshev_nd PRIVATE nda::nda_c)
target_include_directories(chebyshev_nd PRIVATE ${nda_SOURCE_DIR}/include)
target_compile_features(chebyshev_nd PRIVATE cxx_std_20)

add_executable(chebyshev_1d cheb1d.cpp)
target_compile_features(chebyshev_1d PRIVATE cxx_std_20)
target_include_directories(chebyshev_1d PRIVATE cheb)
target_link_libraries(chebyshev_1d PRIVATE nanobench::nanobench xsimd)
target_compile_options(chebyshev_1d PRIVATE $<$<CONFIG:Debug>:-fsanitize=address;-fno-omit-frame-pointer>)
target_link_options(chebyshev_1d PRIVATE $<$<CONFIG:Debug>:-fsanitize=address>)
add_executable(bench1d_1d bench1d.cpp)
target_compile_features(bench1d_1d PRIVATE cxx_std_20)
target_include_directories(bench1d_1d PRIVATE cheb)
target_link_libraries(bench1d_1d PRIVATE nanobench::nanobench xsimd)
target_compile_options(bench1d_1d PRIVATE -march=native)


add_executable(bench_many bench_many.cpp)
target_compile_features(bench_many PRIVATE cxx_std_20)
target_include_directories(bench_many PRIVATE cheb)
target_link_libraries(bench_many PRIVATE nanobench::nanobench xsimd)
target_compile_options(bench_many PRIVATE -march=native)

add_executable(chebgrid_1d cheb_grid_1d.cpp)
target_compile_features(chebgrid_1d PRIVATE cxx_std_20)

add_subdirectory(examples)

add_executable(bench_estrin bench_estrin.cpp)
target_compile_features(bench_estrin PRIVATE cxx_std_20)
target_compile_options(bench_estrin PRIVATE -march=native)
