cmake_minimum_required(VERSION 3.10)
project(LatentDiffusionNLP)

# Require C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable target (replace main.cpp with your actual source files)
add_executable(LatentDiffusionNLP
    main.cpp
    ClassicalDiT/Diffusion_model.cpp
    # Add other source files here as needed
)

# Include directories for headers if needed
target_include_directories(LatentDiffusionNLP PRIVATE
    ${PROJECT_SOURCE_DIR}/ClassicalDiT
    # Add other include directories here
)

# Add any necessary compile options or definitions here
# target_compile_options(LatentDiffusionNLP PRIVATE -Wall -Wextra)