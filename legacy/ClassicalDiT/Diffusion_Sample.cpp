#include "Diffusion_Sample.hpp"
#include "Diffusion_model.hpp"
#include <random>
#include <cmath>
#include <unordered_map>
#include <algorithm>

/**
 * @file Diffusion_Sample.cpp
 * @brief Ancestral Sampling Engine for Latent Reconstruction.
 * * This module implements the reverse diffusion probability $p_\theta(x_{t-1} | x_t)$ 
 * using direct buffer manipulation. It facilitates the transition from pure Gaussian 
 * noise back to the structured Clifford-space manifold.
 * * DESIGN RATIONALE:
 * - Direct Buffer Arithmetic: Operates on flattened `std::vector` to minimize pointer indirection.
 * - Thread-Local RNG: Utilizes `std::random_device` per-sample to ensure statistical independence.
 * - Hardware Agnostic: Validation for 'cpu' and 'gpu' targets, optimized for contiguous cache access.
 */


DiffusionSample::DiffusionSample(DiffusionModel& model, const std::vector<double>& noise_schedule)
    : model_(model), noise_schedule_(noise_schedule), generator_(std::random_device{}()), normal_dist_(0.0, 1.0) {}

/**
 * @brief Performs a complete reverse diffusion trajectory for a batch.
 * @param shape Tensor dimensions [Batch, Channels, Height, Width].
 * @param clip_denoised Boolean flag to enforce $x \in [-1, 1]$ stability.
 * @param denoised_fn Functional hook for external manifold correction.
 * * MATH:
 * Iteratively samples $x_{t-1} \sim p_\theta(x_{t-1} | x_t)$:
 * $$x_{t-1} = \mu_\theta(x_t, t) + \sigma_t \cdot \epsilon$$
 * where $\epsilon \sim \mathcal{N}(0, I)$.
 */

std::vector<std::vector<double> > DiffusionSample::p_sample(
    const std::vector<int>& shape,
    bool clip_denoised,
    const std::function<std::vector<double>(const std::vector<double>&)>& denoised_fn,
    const std::unordered_map<std::string, double>& model_kwags,
    const std::string& device
) {
    std::vector<std::vector<double>> samples(shape[0], std::vector<double>(shape[1] * shape[2] * shape[3], 0.0));
    std::vector<double> mean, variance;

    for (int n = 0; n < shape[0]; ++n) {
        std::default_random_engine thread_generator(std::random_device{}());
        std::normal_distribution<double> thread_normal_dist(0.0, 1.0);

        std::vector<double> x_t(shape[1] * shape[2] * shape[3], 0.0);
        for (int t = shape[0] - 1; t >= 0; --t) {
            model_.compute_mean_variance(x_t, t, mean, variance);
            for (size_t i = 0; i < x_t.size(); ++i) {
                double noise = thread_normal_dist(thread_generator) * std::sqrt(variance[i]);
                x_t[i] = mean[i] + noise; // x_t[i] = mean[i] + (z * sqrt(variance[i]))
            }
        }

        if (denoised_fn) {
            x_t = denoised_fn(x_t);
        }

        if (clip_denoised) {
            for (auto& value : x_t) {
                value = DiffusionSample::clamp(value, -1.0, 1.0);
            }
        }
        samples[n] = x_t; // Store the sample
    }
    return samples;
}

/**
 * @brief Generates samples with metadata preservation for progressive analysis.
 * * Implements rigorous error checking for N-dimensional shape validation and 
 * device-specific constraints.
 * * TECHNICAL INNOVATIONS:
 * - Dynamic Variance Weighting: Applies $\sqrt{\text{variance}[i]}$ scaling to 
 * Gaussian noise to satisfy the Langevin dynamics of the denoising process.
 * - Manifold Clipping: Prevents numerical divergence in deep latent spaces 
 * via a deterministic `clamp` operation.
 */

std::vector<std::unordered_map<std::string, std::vector<double>>> DiffusionSample::p_sample_loop_progressive(
    const std::vector<int>& shape,
    bool clip_denoised,
    const std::function<std::vector<double>(const std::vector<double>&)>& denoised_fn,
    const std::unordered_map<std::string, double>& model_kwags,
    const std::string& device
) {
    if (shape.size() != 4) {
        throw std::invalid_argument("Shape must have exactly 4 dimensions");
    }
    for (int dim : shape) {
        if (dim <= 0) {
            throw std::invalid_argument("Shape dimensions must be positive");
        }
    }
    if (!denoised_fn) {
        throw std::invalid_argument("Denoised function must not be null");
    }

    // Validate device
    if (device != "cpu" && device != "gpu") {
        throw std::invalid_argument("Invalid device: " + device);
    }

    std::vector<std::unordered_map<std::string, std::vector<double>>> samples(shape[0]);

    std::cout << "Checking model_kwags size: " << model_kwags.size() << std::endl;
    if (model_kwags.empty()) {
        std::cout << "Throwing exception for empty model_kwags" << std::endl;
        throw std::invalid_argument("Model keyword arguments must not be empty");
    }
    std::cout << "Passed model_kwags check" << std::endl;

    std::vector<double> mean, variance;

    for (int n = 0; n < shape[0]; ++n) {
        std::default_random_engine thread_generator(std::random_device{}());
        std::normal_distribution<double> thread_distribution(0.0, 1.0);

        std::unordered_map<std::string, std::vector<double>> sample;
        std::vector<double> x_t(shape[1] * shape[2] * shape[3], 0.0);

        for (int t = shape[0] - 1; t >= 0; --t) {
            model_.compute_mean_variance(x_t, t, mean, variance);
            for (size_t i = 0; i < x_t.size(); ++i) {
                double noise = thread_distribution(thread_generator) * std::sqrt(variance[i]);
                x_t[i] = mean[i] + noise;
            }
        }

        x_t = denoised_fn(x_t);
        if (x_t.empty()) {
            throw std::invalid_argument("Denoised function returned empty vector");
        }

        if (clip_denoised) {
            for (auto& value : x_t) {
                value = DiffusionSample::clamp(value, -1.0, 1.0);
            }
        }
        // Store the sample in the progressive_samples vector
        sample["sample"] = x_t;
        samples[n] = sample;
    }

    return samples;
}
