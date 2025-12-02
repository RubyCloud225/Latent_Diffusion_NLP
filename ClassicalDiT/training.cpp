#include <iostream>
#include <vector>
#include <stdexcept>
#include "BetaSchedule.hpp"
#include "GaussianDiffusion.hpp"


// Assume all your classes are declared and included:
// BetaSchedule, GaussianDiffusion, Multivector, etc.

struct Multivector {
    std::vector<float> scalars_fp32;
    std::vector<uint16_t> scalars_fp16;
    std::vector<float> scalars_fp32_full;
};

// Convert compressed data into vector<vector<double>> suitable for GaussianDiffusion training
std::vector<std::vector<double>> convert_to_training_vectors(const std::vector<Multivector>& compressed) {
    std::vector<std::vector<double>> training_data;
    training_data.reserve(compressed.size());
    for (const auto& mv : compressed) {
        std::vector<double> sample;
        sample.reserve(mv.scalars_fp32_full.size());
        for (float f : mv.scalars_fp32_full) {
            sample.push_back(static_cast<double>(f));
        }
        training_data.push_back(std::move(sample));
    }
    return training_data;
}

// Updated train function integrating BetaSchedule loss calculation
void train_diffusion_model_with_beta_schedule(const std::vector<Multivector>& compressed,
                                              int epochs,
                                              int num_timesteps,
                                              double beta_start,
                                              double beta_end) {
    if (compressed.empty()) {
        std::cerr << "Compressed data is empty. Aborting training.\n";
        return;
    }

    auto training_data = convert_to_training_vectors(compressed);

    BetaSchedule beta_schedule(epochs, beta_start);
    GaussianDiffusion diffusion(num_timesteps, beta_start, beta_end);

    std::cout << "Starting training with BetaSchedule-managed losses...\n";

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Train for one epoch: diffusion.train should run forward/backward
        diffusion.train(training_data, 1);

        // After training, BetaSchedule calculates losses and updates beta
        // Pass diffusion model outputs or data needed for loss calculation to BetaSchedule here

        // For example, you might collect model predictions and true values in diffusion.train
        // But since BetaSchedule handles loss calc internally, just call update with empty placeholders:

        std::vector<double> nll_losses;    // BetaSchedule internally calculates these
        std::vector<double> entropy_losses; // BetaSchedule internally calculates these

        double updated_beta = beta_schedule.update(nll_losses, entropy_losses, epoch);

        std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                  << " Beta updated to: " << updated_beta << std::endl;

        // Optionally adjust diffusion parameters with updated_beta if your diffusion code supports it
    }

    std::cout << "Training complete.\n";
}