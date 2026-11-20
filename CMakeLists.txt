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
