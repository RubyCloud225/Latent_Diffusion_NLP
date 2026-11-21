# Apple-only experimental Metal fixture, moved unchanged in function from root.
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
    test_clifford_projection_metal.cpp
    "${PROJECT_SOURCE_DIR}/src/backend/metal/CliffordLatentMetal.mm"
)
add_dependencies(test_clifford_projection_metal clifford_metallib)
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
add_test(NAME clifford_projection_metal COMMAND test_clifford_projection_metal)
set_tests_properties(clifford_projection_metal PROPERTIES LABELS "experimental;metal")
