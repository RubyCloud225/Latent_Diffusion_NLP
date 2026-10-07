cmake_minimum_required(VERSION 3.16)
project(LatentDiffusionNLP)

# Require C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)

add_library(latent_experimental STATIC
    src/experimental/CliffordProjection.cpp
    src/nn/Adam.cpp
)

target_include_directories(latent_experimental PUBLIC
    ${PROJECT_SOURCE_DIR}/latent
)

if(BUILD_TESTING)
    add_executable(test_fp16
        tests/core/test_fp16.cpp
    )
    target_include_directories(test_fp16 PRIVATE
        ${PROJECT_SOURCE_DIR}/latent
    )
    add_test(NAME fp16 COMMAND test_fp16)

    add_executable(test_clifford_projection
        tests/experimental/test_clifford_projection.cpp
    )
    target_link_libraries(test_clifford_projection PRIVATE
        latent_experimental
    )
    add_test(NAME clifford_projection COMMAND test_clifford_projection)
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
endif()
