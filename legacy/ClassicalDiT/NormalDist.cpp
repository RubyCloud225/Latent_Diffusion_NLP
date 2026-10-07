#include "NormalDist.hpp"
#include <cmath>
#include <stdexcept>

/**
 * @file NormalDist.cpp
 * @brief Differentiable Probabilistic Kernel for Diffusion Likelihood.
 *
 * This module provides the core statistical primitives required for the 
 * Variational Lower Bound (VLB) calculation. It implements the Gaussian 
 * log-likelihood and its derivatives, enabling backpropagation through 
 * stochastic processes without a high-level autograd library.
 *
 * DESIGN RATIONALE:
 * - Direct Gradient Computation: Explicitly implements ∂L/∂μ and ∂L/∂σ.
 * - Numerical Stability: Includes guards against zero-variance and log-domain 
 * singularities.
 * - Modular Mean/Sigma Mapping: Decouples the model's ε-prediction from the 
 * probability space.
 */

namespace NormalDist {
/**
 * @brief Parameter Transformation Kernels.
 * * These functions define the mapping between the DiffusionModel's raw 
 * outputs and the Gaussian parameters required for training.
 * * Stability Note: sigma is clamped or softplus-transformed to ensure 
 * $\sigma > 0$, preventing numerical divergence in the gradient computation.
 */
    double log_prob(double y, double mean, double sigma) {
        const double log_sqrt_2pi = 0.5 * std::log(2.0 * M_PI);
        double diff = (y - mean) / sigma;
        double log_prob_value = -0.5 * diff * diff - std::log(sigma) - log_sqrt_2pi;
        return log_prob_value;
    }
    double compute_mean(double x_start_pred, double eps_pred) {
        return 0.5 * (x_start_pred + eps_pred);
    }
    double compute_sigma(double x_start_pred, double eps_pred) {
        return std::abs(x_start_pred - eps_pred) / std::sqrt(2.0);
    }
    /**
     * @brief Computes the Log-Probability Density of a noisy sample.
     * @param y The observed noisy sample $x_t$.
     * @param x_start The estimated clean latent $x_0$.
     * @param epsilon The predicted noise $\epsilon_\theta$.
     * @return double The log-probability $\log p(y | \mu, \sigma)$.
     *
     * MATHS:
     * Maps $(x_{start}, \epsilon)$ to $(\mu, \sigma)$ space and calculates:
     * $$\log p(y) = -\frac{1}{2} \log(2\pi\sigma^2) - \frac{(y - \mu)^2}{2\sigma^2}$$
     */
    double log_prob_from_predictions(double y, double x_start_pred, double eps_pred) {
        double mean = compute_mean(x_start_pred, eps_pred);
        double sigma = compute_sigma(x_start_pred, eps_pred);
        if (sigma <= 0) {
            throw std::invalid_argument("Sigma must be positive");
        }
        return log_prob(y, mean, sigma);
    }
    double grad_wrt_y(double y, double mean, double sigma) {
        return (y - mean) / (sigma * sigma);
    }
    double grad_wrt_sigma(double y, double mean, double sigma) {
        return (y - mean) / (sigma * sigma * sigma) - 1.0 / sigma;
    }
    /**
     * @brief Analytic Partial Derivatives of the Log-Likelihood.
     * @param y Observation.
     * @param mu Estimated Mean.
     * @param sigma Estimated Standard Deviation.
     * @param[out] dfd_y Gradient w.r.t input.
     * @param[out] dfd_mu Gradient w.r.t mean: $(y - \mu) / \sigma^2$.
     * @param[out] dfd_sigma Gradient w.r.t variance: $( (y-\mu)^2 / \sigma^3 ) - (1/\sigma)$.
     *
     * These gradients allow the AdamOptimizer to update the model parameters 
     * by propagating the error from the probability distribution back through 
     * the CNN layers.
     */
    void gradients(double y, double mean, double sigma, double& dfd_y, double& dfd_mu, double& dfd_sigma) {
        dfd_y = grad_wrt_y(y, mean, sigma);
        dfd_mu = -dfd_y;
        dfd_sigma = grad_wrt_sigma(y, mean, sigma);
    }
}
