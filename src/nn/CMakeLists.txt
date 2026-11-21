target_sources(latent_core PRIVATE
    Linear.cpp
    Activation.cpp
    Conv2D.cpp
    AdamFP64.cpp
)

# Experimental FP32 Adam is deliberately not part of latent_core.
target_sources(latent_experimental PRIVATE
    Adam.cpp
)

add_subdirectory(training)
