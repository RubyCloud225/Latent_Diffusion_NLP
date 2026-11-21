# Library targets are owned here; subdirectories own their source lists.
add_library(latent_core STATIC)
target_include_directories(latent_core PUBLIC "${PROJECT_SOURCE_DIR}")
target_compile_features(latent_core PUBLIC cxx_std_17)

# Retain the legacy experimental include contract.
add_library(latent_experimental STATIC)
target_include_directories(latent_experimental PUBLIC "${PROJECT_SOURCE_DIR}/latent")
target_compile_features(latent_experimental PUBLIC cxx_std_17)

add_subdirectory(core)
add_subdirectory(diffusion)
add_subdirectory(nn)
add_subdirectory(experimental)
