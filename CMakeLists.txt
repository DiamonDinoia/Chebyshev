cmake_minimum_required(VERSION 3.15)
project(ChebyshevND LANGUAGES CXX)


# ─────────────────────────────────────────────
# Use CPM to fetch nda
include(FetchContent)

FetchContent_Declare(
  CPM
  GIT_REPOSITORY https://github.com/cpm-cmake/CPM.cmake
  GIT_TAG v0.41.0
)

FetchContent_MakeAvailable(CPM)

CPMAddPackage(
  NAME nda
  GITHUB_REPOSITORY DiamonDinoia/nda
  GIT_TAG refactor-cmake
)

# ─────────────────────────────────────────────
# Build executable
add_executable(chebyshev_nd main.cpp)

target_link_libraries(chebyshev_nd PRIVATE nda::nda_c)
target_include_directories(chebyshev_nd PRIVATE ${nda_SOURCE_DIR}/include)
target_compile_features(chebyshev_nd PRIVATE cxx_std_17)