# Test names and CTest labels are stable public development interfaces.
add_subdirectory(core)
add_subdirectory(diffusion)
add_subdirectory(nn)
add_subdirectory(experimental)

if(APPLE AND BUILD_METAL_TESTS)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
    add_subdirectory(backend/metal)
endif()
