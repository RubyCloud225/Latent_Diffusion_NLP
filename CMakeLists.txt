cmake_minimum_required(VERSION 3.16)
project(LatentDiffusionNLP)

# Require C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)

# P0.4: the production Tensor implementation belongs to latent_core,
# not to the isolated experimental Clifford library.
add_library(latent_core STATIC
    src/core/Tensor.cpp
    src/core/Random.cpp
    src/diffusion/NoiseSchedule.cpp
    src/diffusion/GaussianDiffusion.cpp
    src/nn/Linear.cpp
    src/nn/Activation.cpp
    src/nn/Conv2D.cpp
    src/nn/AdamFP64.cpp
)
target_include_directories(latent_core PUBLIC
    ${PROJECT_SOURCE_DIR}
)
target_compile_features(latent_core PUBLIC cxx_std_17)

add_library(latent_experimental STATIC
    src/experimental/CliffordProjection.cpp
    src/nn/Adam.cpp
)
target_include_directories(latent_experimental PUBLIC
    ${PROJECT_SOURCE_DIR}/latent
)
target_compile_features(latent_experimental PUBLIC cxx_std_17)

if(BUILD_TESTING)
    add_executable(test_fp16
        tests/core/test_fp16.cpp
    )
    target_include_directories(test_fp16 PRIVATE
        ${PROJECT_SOURCE_DIR}/latent
    )
    add_test(NAME fp16 COMMAND test_fp16)
    set_tests_properties(fp16 PROPERTIES LABELS "p0;core")

    # P0.3: reference fixture and runner only. The production diffusion API
    # is implemented and tested in P0.6-P0.9.
    add_executable(test_diffusion_harness
        tests/diffusion/test_diffusion_harness.cpp
    )
    add_test(NAME diffusion_harness COMMAND test_diffusion_harness)
    set_tests_properties(diffusion_harness PROPERTIES LABELS "p0;diffusion")

    add_executable(test_clifford_projection
        tests/experimental/test_clifford_projection.cpp
    )
    target_link_libraries(test_clifford_projection PRIVATE
        latent_experimental
    )
    add_test(NAME clifford_projection COMMAND test_clifford_projection)
    set_tests_properties(clifford_projection PROPERTIES LABELS "experimental")

    # P0.4: validate the production Tensor through the core library.
    add_executable(test_tensor
        tests/core/test_tensor.cpp
    )
    target_link_libraries(test_tensor PRIVATE latent_core)
    add_test(NAME tensor COMMAND test_tensor)
    set_tests_properties(tensor PROPERTIES LABELS "p0;core")

    # P0.5 — Seeded random-number generator.
    add_executable(test_random
        tests/core/test_random.cpp
    )

    target_link_libraries(test_random PRIVATE
        latent_core
    )

    add_test(NAME random COMMAND test_random)

    set_tests_properties(random PROPERTIES
        LABELS "p0;core"
    )

    # P0.6 — Production diffusion noise schedule.
    add_executable(test_noise_schedule
        tests/diffusion/test_noise_schedule.cpp
    )

    target_link_libraries(test_noise_schedule PRIVATE
        latent_core
    )

    add_test(
        NAME noise_schedule
        COMMAND test_noise_schedule
    )

    set_tests_properties(noise_schedule PROPERTIES
        LABELS "p0;diffusion"
    )

    # P0.7 — Production forward diffusion.
    add_executable(test_forward_diffusion
        tests/diffusion/test_forward_diffusion.cpp
    )
    target_link_libraries(test_forward_diffusion PRIVATE latent_core)
    add_test(NAME forward_diffusion COMMAND test_forward_diffusion)
    set_tests_properties(forward_diffusion PROPERTIES LABELS "p0;diffusion")

    # P0.8 — Deterministic reverse diffusion mathematics.
    add_executable(test_reverse_diffusion
        tests/diffusion/test_reverse_diffusion.cpp
    )
    target_link_libraries(test_reverse_diffusion PRIVATE latent_core)
    add_test(NAME reverse_diffusion COMMAND test_reverse_diffusion)
    set_tests_properties(reverse_diffusion PROPERTIES LABELS "p0;diffusion")

    # NN.1.01 — Neural-network Layer and Parameter interfaces.
    add_executable(test_nn_layer
        tests/nn/test_layer.cpp
    )

    target_link_libraries(test_nn_layer PRIVATE
        latent_core
    )

    add_test(NAME nn_layer COMMAND test_nn_layer)

    set_tests_properties(nn_layer PROPERTIES
        LABELS "nn;core"
    )
    
    # NN.1.03A — FP64 ReLU activation.
    add_executable(test_nn_relu tests/nn/test_relu.cpp)
    target_link_libraries(test_nn_relu PRIVATE latent_core)
    add_test(NAME nn_relu COMMAND test_nn_relu)
    set_tests_properties(nn_relu PROPERTIES LABELS "nn")

    # NN.1.03B — FP64 SiLU activation.
    add_executable(test_nn_silu tests/nn/test_silu.cpp)
    target_link_libraries(test_nn_silu PRIVATE latent_core)
    add_test(NAME nn_silu COMMAND test_nn_silu)
    set_tests_properties(nn_silu PROPERTIES LABELS "nn")

    # NN.1.03C — Independent ReLU / SiLU numerical-gradient validation.
    add_executable(test_nn_activation_validation
        tests/nn/test_activation_validation.cpp
    )
    target_link_libraries(test_nn_activation_validation PRIVATE latent_core)
    add_test(NAME nn_activation_validation COMMAND test_nn_activation_validation)
    set_tests_properties(nn_activation_validation PROPERTIES LABELS "nn")

    # NN.1.02 — FP64 Linear Layer.
    add_executable(test_nn_linear
        tests/nn/test_linear.cpp
    )

    target_link_libraries(test_nn_linear PRIVATE
        latent_core
    )

    add_test(NAME nn_linear COMMAND test_nn_linear)

    set_tests_properties(nn_linear PROPERTIES
        LABELS "nn;core"
    )

    # NN.1.04 — Conv2D forward propagation.
    add_executable(test_nn_conv2d_forward
        tests/nn/test_conv2d_forward.cpp
    )

    target_link_libraries(
        test_nn_conv2d_forward
        PRIVATE latent_core
    )

    add_test(
        NAME nn_conv2d_forward
        COMMAND test_nn_conv2d_forward
    )

    set_tests_properties(
        nn_conv2d_forward
        PROPERTIES LABELS "nn"
    )

    # NN.1.05 — Conv2D backward propagation.
    add_executable(test_nn_conv2d_backward
        tests/nn/test_conv2d_backward.cpp
    )

    target_link_libraries(
        test_nn_conv2d_backward
        PRIVATE latent_core
    )

    add_test(
        NAME nn_conv2d_backward
        COMMAND test_nn_conv2d_backward
    )

    set_tests_properties(
        nn_conv2d_backward
        PROPERTIES LABELS "nn"
    )

    # NN.1.06 — Production FP64 Adam optimiser.
    add_executable(test_nn_adam_fp64
        tests/nn/test_adam_fp64.cpp
    )

    target_link_libraries(
        test_nn_adam_fp64
        PRIVATE latent_core
    )

    add_test(
        NAME nn_adam_fp64
        COMMAND test_nn_adam_fp64
    )

    set_tests_properties(
        nn_adam_fp64
        PROPERTIES LABELS "nn"
    )

endif()

option(BUILD_METAL_TESTS "Build tests requiring the Xcode Metal Toolchain" ${APPLE})

if(APPLE AND BUILD_TESTING AND BUILD_METAL_TESTS)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

    set(CLIFFORD_METAL_SOURCE
        "${PROJECT_SOURCE_DIR}/src/backend/metal/CliffordLatent.metal"
    )
    set(CLIFFORD_METAL_AIR
        "${CMAKE_BINARY_DIR}/CliffordProjection.air"
    )
    set(CLIFFORD_METALLIB
        "${CMAKE_BINARY_DIR}/CliffordProjection.metallib"
    )

    add_custom_command(
        OUTPUT "${CLIFFORD_METAL_AIR}"
        COMMAND xcrun -sdk macosx metal
            -c "${CLIFFORD_METAL_SOURCE}"
            -o "${CLIFFORD_METAL_AIR}"
        DEPENDS "${CLIFFORD_METAL_SOURCE}"
        COMMENT "Compiling CliffordProjection Metal shader"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${CLIFFORD_METALLIB}"
        COMMAND xcrun -sdk macosx metallib
            "${CLIFFORD_METAL_AIR}"
            -o "${CLIFFORD_METALLIB}"
        DEPENDS "${CLIFFORD_METAL_AIR}"
        COMMENT "Building CliffordProjection.metallib"
        VERBATIM
    )
    add_custom_target(clifford_metallib ALL
        DEPENDS "${CLIFFORD_METALLIB}"
    )

    add_executable(test_clifford_projection_metal
        tests/backend/metal/test_clifford_projection_metal.cpp
        src/backend/metal/CliffordLatentMetal.mm
    )
    add_dependencies(test_clifford_projection_metal
        clifford_metallib
    )
    target_include_directories(test_clifford_projection_metal PRIVATE
        "${PROJECT_SOURCE_DIR}"
    )
    target_compile_definitions(test_clifford_projection_metal PRIVATE
        CLIFFORD_METALLIB_PATH="${CLIFFORD_METALLIB}"
    )
    target_compile_options(test_clifford_projection_metal PRIVATE
        $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
    )
    target_link_libraries(test_clifford_projection_metal PRIVATE
        latent_experimental
        "-framework Foundation"
        "-framework Metal"
    )
    add_test(NAME clifford_projection_metal
        COMMAND test_clifford_projection_metal
    )
    set_tests_properties(clifford_projection_metal PROPERTIES LABELS "experimental;metal")
endif()

# Optional API documentation: normal configure/build/test does not require Doxygen.
option(BUILD_DOCS "Enable Doxygen API documentation target" OFF)
if(BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/docs")
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_GENERATE_LATEX NO)
    set(DOXYGEN_EXTRACT_ALL NO)
    set(DOXYGEN_WARN_IF_UNDOCUMENTED YES)
    set(DOXYGEN_QUIET NO)
    doxygen_add_docs(docs
        "${PROJECT_SOURCE_DIR}/latent/core"
        "${PROJECT_SOURCE_DIR}/latent/diffusion"
        "${PROJECT_SOURCE_DIR}/latent/nn"
        COMMENT "Generate Doxygen production-core API documentation"
    )
endif()
