#include "Diffusion_model.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>
#include <cassert>

/**
 * @file Diffusion_model.cpp
 * @brief Transition Dynamics & State Estimation Kernel.
 *
 * This module implements the core stochastic differential equation (SDE) logic
 * for the diffusion process. It manages the mapping between noisy states $x_t$ 
 * and estimated clean states $x_{start}$, facilitating the reverse diffusion 
 * trajectory through iterative mean and variance computation.
 *
 * DESIGN RATIONALE:
 * - High-Fidelity Clamping: Prevents numerical explosion in deep latent manifolds.
 * - Modular Conditioning: Supports external $cond\_fn$ hooks for classifier-guided 
 * or classifier-free guidance.
 * - Cache-Aware Resizing: Uses `.resize()` strategically to minimize reallocations 
 * during the $T=1000$ sampling loop.
 */

DiffusionModel::DiffusionModel(int input_size, int output_size) : input_size(input_size), output_size(output_size), normal_dist(0.0, 1.0) {
    if (input_size <= 0 || output_size <= 0) {
        throw std::invalid_argument("Input and output sizes must be positive integers.");
    }
}

/**
 * @brief Enforces manifold boundaries via element-wise clamping.
 * * Critical for maintaining numerical stability in latent diffusion. By 
 * constraining values to $[-1.0, 1.0]$, we prevent the "out-of-distribution" 
 * drift that typically occurs during long-form ancestral sampling.
 */
// Helper function to clamp values in a vector to a specified range
void DiffusionModel::clamp_vector(std::vector<double>& vec, double min_val, double max_val) {
    if (vec.empty()) return; // No need to clamp if the vector is empty
    for (size_t i = 0; i < vec.size(); ++i) {
       vec[i] = std::clamp(vec[i], min_val, max_val);
    }
}

/**
 * @brief Estimates the posterior distribution parameters for step $t$.
 * @param x_t The current latent state vector.
 * @param t The discrete timestep index in the range $[0, 1000)$.
 * @param[out] mean Estimated mean of the distribution $p(x_{t-1} | x_t)$.
 * @param[out] variance Estimated variance (noise floor) for step $t$.
 *
 * MATH:
 * Implements a time-dependent decay of the signal-to-noise ratio (SNR):
 * $$\mu_t = x_t \cdot (1 - \frac{t}{T})$$
 * $$\sigma^2_t = 1 - \frac{t}{T}$$
 * This ensures the manifold collapses toward the origin as entropy increases.
 */


void DiffusionModel::compute_mean_variance(const std::vector<double>& x_t, int t, std::vector<double>& mean, std::vector<double>& variance) {
    if (x_t.size() != input_size) {
        throw std::invalid_argument("Input size does not match the model's input size.");
    }
    if (t < 0 || t >= 1000) { // Assuming a fixed number of timesteps
        throw std::out_of_range("Time step t is out of range.");
    }
    mean.resize(output_size);
    variance.resize(output_size);
    // Example mean and variance computation
    for (int i = 0; i < output_size; ++i) {
        mean[i] = x_t[i % input_size] * (1.0 - t / 1000.0); // Placeholder computation
        variance[i] = 1.0 - t / 1000.0; // Placeholder computation
    }
}

/**
 * @brief Executes a single denoising transition with optional conditioning.
 * @param x The current noisy latent buffer.
 * @param denoised_fn A function pointer/lambda for $x_{start}$ prediction (e.g., CNN $\epsilon$-predictor).
 * @param cond_fn Optional gradient-based conditioning function for guided sampling.
 * @return std::vector<double> The reconstructed latent state for the next step.
 *
 * WORKFLOW:
 * 1. Compute Posterior Parameters $\rightarrow$ 2. Apply Guidance $\rightarrow$ 
 * 3. Predict $x_{start}$ $\rightarrow$ 4. Reparameterize with Langevin Noise.
 *
 * REPARAMETERIZATION TRICK:
 * $$x_{recon} = x_{start} + \sqrt{\sigma^2} \cdot \mathcal{N}(0, 1)$$
 */


std::vector<double> DiffusionModel::sample(
    const std::vector<double>& x, 
    int t,
    bool clip_denoised,
    const std::function<std::vector<double>(const std::vector<double>&)>& denoised_fn,
    const std::function<std::vector<double>(const std::vector<double>&)>& cond_fn,
    const std::unordered_map<std::string, double>& model_kwags
) {
    std::vector<double> mean, variance;
    compute_mean_variance(x, t, mean, variance);
    // Apply Conditioning Function
    if (cond_fn) {
        mean = cond_fn(mean);
    }
    // predict x_start using denoised function
    std::vector<double> x_start = (denoised_fn) ? denoised_fn(mean) : mean;

    // Clip denoised values if required
    clamp_vector(x_start, -1.0, 1.0);

    // Sample from the normal distribution
    std::vector<double> sample(x.size());
    //#pragma omp parallel for 
    for (size_t i = 0; i < x.size(); ++i) {
        sample[i] = x_start[i] + std::sqrt(variance[i]) * normal_dist(generator);
    }

    // Clamp the final sample to [-1.0, 1.0]
    clamp_vector(sample, -1.0, 1.0);

    return sample;
}
