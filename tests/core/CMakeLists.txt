# FP16 exercises the legacy header-only API and uses its historical include root.
add_executable(test_fp16 test_fp16.cpp)
target_include_directories(test_fp16 PRIVATE "${PROJECT_SOURCE_DIR}/latent")
add_test(NAME fp16 COMMAND test_fp16)
set_tests_properties(fp16 PROPERTIES LABELS "p0;core")

add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE latent_core)
add_test(NAME tensor COMMAND test_tensor)
set_tests_properties(tensor PROPERTIES LABELS "p0;core")

add_executable(test_random test_random.cpp)
target_link_libraries(test_random PRIVATE latent_core)
add_test(NAME random COMMAND test_random)
set_tests_properties(random PROPERTIES LABELS "p0;core")
