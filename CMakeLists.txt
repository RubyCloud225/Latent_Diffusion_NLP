cmake_minimum_required(VERSION 3.16)
project(LatentDiffusionNLP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

option(BUILD_METAL_TESTS "Build tests requiring the Xcode Metal toolchain" ${APPLE})
option(BUILD_DOCS "Enable Doxygen API documentation target" OFF)

# Libraries and their source files are owned by src/ subdirectories.
add_subdirectory(src)

# Test names, fixtures and labels are owned by tests/ subdirectories.
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# Optional API documentation does not affect normal builds.
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
