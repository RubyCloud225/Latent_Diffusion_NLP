# Keep the original P0 test names and labels.
add_executable(test_diffusion_harness test_diffusion_harness.cpp)
add_test(NAME diffusion_harness COMMAND test_diffusion_harness)
set_tests_properties(diffusion_harness PROPERTIES LABELS "p0;diffusion")

function(add_diffusion_test target source ctest_name)
    add_executable(${target} "${source}")
    target_link_libraries(${target} PRIVATE latent_core)
    add_test(NAME ${ctest_name} COMMAND ${target})
    set_tests_properties(${ctest_name} PROPERTIES LABELS "p0;diffusion")
endfunction()

add_diffusion_test(test_noise_schedule test_noise_schedule.cpp noise_schedule)
add_diffusion_test(test_forward_diffusion test_forward_diffusion.cpp forward_diffusion)
add_diffusion_test(test_reverse_diffusion test_reverse_diffusion.cpp reverse_diffusion)
