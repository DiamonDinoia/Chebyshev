add_executable(simple_eval simple_eval.cpp)
target_compile_features(simple_eval PRIVATE cxx_std_20)
target_link_libraries(simple_eval PRIVATE cheb)

add_executable(const_eval const_eval.cpp)
target_compile_features(const_eval PRIVATE cxx_std_17)
target_link_libraries(const_eval PRIVATE cheb)


add_executable(eps_eval eps_eval.cpp)
target_compile_features(eps_eval PRIVATE cxx_std_20)
target_link_libraries(eps_eval PRIVATE cheb)



add_executable(es_kernel es_kernel.cpp)
target_compile_features(es_kernel PRIVATE cxx_std_20)
target_link_libraries(es_kernel PRIVATE cheb)


add_executable(many_funcs many_func_example.cpp)
target_compile_features(many_funcs PRIVATE cxx_std_20)
target_link_libraries(many_funcs PRIVATE cheb)

add_executable(many_funcs_pts many_func_many_pts_example.cpp)
target_compile_features(many_funcs_pts PRIVATE cxx_std_20)
target_link_libraries(many_funcs_pts PRIVATE cheb)
