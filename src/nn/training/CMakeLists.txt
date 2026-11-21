# Shared FP64 training foundation. Add new training sources here.
target_sources(latent_core PRIVATE
    TrainingPolicy.cpp
    Initialisation.cpp
    GradientController.cpp
)
