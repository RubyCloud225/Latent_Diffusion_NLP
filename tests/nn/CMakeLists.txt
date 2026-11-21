# A test compiles only its test source; implementations are linked from latent_core.
function(add_nn_test target source ctest_name)
    add_executable(${target} "${source}")
    target_link_libraries(${target} PRIVATE latent_core)
    add_test(NAME ${ctest_name} COMMAND ${target})
    set_tests_properties(${ctest_name} PROPERTIES LABELS "nn")
endfunction()

add_nn_test(test_nn_layer test_layer.cpp nn_layer)
add_nn_test(test_nn_relu test_relu.cpp nn_relu)
add_nn_test(test_nn_silu test_silu.cpp nn_silu)
add_nn_test(test_nn_activation_validation test_activation_validation.cpp nn_activation_validation)
add_nn_test(test_nn_linear test_linear.cpp nn_linear)
add_nn_test(test_nn_conv2d_forward test_conv2d_forward.cpp nn_conv2d_forward)
add_nn_test(test_nn_conv2d_backward test_conv2d_backward.cpp nn_conv2d_backward)
add_nn_test(test_nn_adam_fp64 test_adam_fp64.cpp nn_adam_fp64)
add_nn_test(test_nn_training_policy test_training_policy.cpp nn_training_policy)
add_nn_test(test_initialisation test_initialisation.cpp nn_initialisation)

# NN.1.07C was committed as source/test files but not previously registered.
add_nn_test(test_gradient_controller test_gradient_controller.cpp nn_gradient_controller)

# Preserve the combined core label on these two existing tests.
set_tests_properties(nn_layer nn_linear PROPERTIES LABELS "nn;core")
