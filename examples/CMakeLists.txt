add_executable(simple_eval simple_eval.cpp)
target_compile_features(simple_eval PRIVATE cxx_std_23)
target_link_libraries(simple_eval PRIVATE cheb)

add_executable(const_eval const_eval.cpp)
target_compile_features(const_eval PRIVATE cxx_std_23)
target_link_libraries(const_eval PRIVATE cheb)


add_executable(eps_eval eps_eval.cpp)
target_compile_features(eps_eval PRIVATE cxx_std_23)
target_link_libraries(eps_eval PRIVATE cheb)



add_executable(es_kernel es_kernel.cpp)
target_compile_features(es_kernel PRIVATE cxx_std_23)
target_link_libraries(es_kernel PRIVATE cheb)
