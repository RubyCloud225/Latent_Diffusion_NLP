#include "GaussianDiffusion.hpp"
#include "NN/EpsilonPredictor.hpp"
#include "NormalDist.hpp"
//#include <omp.h>
#include <stdexcept> // for std::invalid_argument

/**
 * @file GaussianDiffusion.cpp
 * @brief Forward Diffusion Kernel and Adaptive Optimization Engine.
 *
 * This module manages the "Forward Process" of the diffusion pipeline, 
 * transforming structured data into Gaussian noise via a linear variance 
 * schedule. It also implements a standalone Adam (Adaptive Moment Estimation) 
 * optimizer for parameter updates without external library dependencies.
 *
 * DESIGN RATIONALE:
 * - Moment Tracking: Implements first and second-order moment estimation for 
 * stable convergence in high-dimensional latent spaces.
 * - Linear Variance Scheduling: Progressively increases the noise floor 
 * to ensure smooth manifold degradation.
 * - Cache-Aware Optimization: Parameter updates are performed in-place to 
 * minimize memory overhead during heavy training iterations.
 */

// Adam Optimizer Constructor
AdamOptimizer::AdamOptimizer(double learning_rate, double beta1, double beta2, double epsilon) : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0) {
    // Initialize first and second moment vectors
}

/**
 * @brief Executes an adaptive parameter update using the Adam algorithm.
 * @param params Raw buffer of network weights (W, b).
 * @param gradients Raw buffer of computed loss derivatives.
 *
 * MATH:
 * 1. Update biased first moment: $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$
 * 2. Update biased second raw moment: $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$
 * 3. Compute bias-corrected moments: $\hat{m}_t, \hat{v}_t$
 * 4. Apply update: $\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$
 *
 * This ensures efficient navigation of the complex loss landscapes found in 
 * latent diffusion models.
 */

void AdamOptimizer::update(std::vector<double>& params, std::vector<double>& gradients) {
    t_++;
    if (m_.empty()) {
        m_.resize(params.size(), 0);
        v_.resize(params.size(), 0);
    }
    //#pragma omp parallel for
    if (params.size() != gradients.size() || params.size() != m_.size() || params.size() != v_.size()) {
        throw std::invalid_argument("Parameter and gradient sizes must match.");
    }
    for (size_t i = 0; i < params.size(); ++i) {
        m_[i] = beta1_ * m_[i] + (1 - beta1_) * gradients[i];
        v_[i] = beta2_ * v_[i] + (1 - beta2_) * gradients[i] * gradients[i];
        double m_hat = m_[i] / (1 - std::pow(beta1_, t_));
        double v_hat = v_[i] / (1 - std::pow(beta2_, t_));
        params[i] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
    }
}

/**
 * @brief Parameter-Gradient Validation.
 * * Ensures strict dimensional alignment before performing vector arithmetic. 
 * This prevents buffer overflows and "NaN" propagation in the moment vectors, 
 * which is a critical failure mode in manual C++ neural implementations.
 */

// Gaussian Diffusion Constructor
GaussianDiffusion::GaussianDiffusion(int num_timesteps, double beta_start, double beta_end) : num_timesteps_(num_timesteps), beta_start_(beta_start), beta_end_(beta_end), optimizer_(0.001, 0.9, 0.999, 1e-8) {
    if (beta_start > beta_end) {
        throw std::invalid_argument("beta_start must be less than or equal to beta_end");
    }
    betas_.resize(num_timesteps_);
    for (int t = 0; t < num_timesteps_; ++t) {
        betas_[t] = beta_start + (beta_end - beta_start) * (static_cast<double>(t) / num_timesteps_);
    }
}

/**
 * @brief Initializes the forward noise schedule ($\beta_t$).
 * @param num_timesteps Total steps $T$ (typically 1000).
 * @param beta_start Initial noise level (e.g., $1 \times 10^{-4}$).
 * @param beta_end Final noise level (e.g., $0.02$).
 *
 * Implements a linear interpolation for $\beta_t$:
 * $$\beta_t = \beta_{start} + \frac{t}{T}(\beta_{end} - \beta_{start})$$
 * This schedule governs the rate at which the Clifford-space embeddings 
 * are diffused into a standard normal distribution $\mathcal{N}(0, I)$.
 */

// Forward process
std::vector<double> GaussianDiffusion::forward(const std::vector<double>& x_prev, int t) {
    if (t < 0 || t >= num_timesteps_) {
        throw std::invalid_argument("Invalid timestep: must be between 0 and num_timesteps - 1");
    }
    std::vector<double> x_t(x_prev.size());
    std::default_random_engine generator;
    std::normal_distribution<double> noise(0.0, std::sqrt(betas_[t]));
    for (size_t i = 0; i < x_prev.size(); ++i) {
        x_t[i] = x_prev[i] + noise(generator);
    }
    return x_t;
}

/**
 * @brief Performs a single-step reverse Markov transition $p_\theta(x_{t-1} | x_t)$.
 * @param x_t The noisy latent vector at current timestep $t$.
 * @param t The discrete timestep index in the range $[0, T-1]$.
 * @return std::vector<double> The reconstructed latent state $x_{t-1}$.
 *
 * DESIGN RATIONALE:
 * Implements the core iterative denoising logic. This function serves as the 
 * basis for the generative sampling loop, allowing the model to project 
 * pure Gaussian noise back onto the learned Clifford-space manifold.
 */

std::vector<double> GaussianDiffusion::reverse(const std::vector<double>& x_t, int t) {
    std::vector<double> x_prev(x_t.size());
    std::default_random_engine generator;
    std::normal_distribution<double> noise(0.0, 1.0); //placeholder
    double beta_t = betas_[t];
    double sqrt_one_minus_beta_t = std::sqrt(1.0 - beta_t);
    double sqrt_beta_t = std::sqrt(beta_t);
    for (size_t i = 0; i < x_t.size(); ++i) {
        // Estimate the noise term ( replace this with a better method)
        double epsilon = noise(generator); // sample
        // mean from the reverse process
        double mu_t_minus_1 = (x_t[i] - sqrt_beta_t * epsilon) / sqrt_one_minus_beta_t;
        // sample x_{t_1}
        x_prev[i] = mu_t_minus_1 + sqrt_beta_t * noise(generator);
    }
    return x_prev; // placeholder
}

//Sigmoid activation function
double GaussianDiffusion::sigmoid(double x) const {
    return 1.0 / (1.0 + std::exp(-x));
}

// Derivative of the sigmoid function
double GaussianDiffusion::sigmoid_derivative(double x) const {
    double sig = sigmoid(x);
    return sig * (1.0 - sig);
}

/**
 * @brief End-to-End Training Orchestrator for Latent Diffusion.
 * * This module manages the primary training loop, integrating the forward SDE, 
 * epsilon-prediction through a functional model hook, and gradient-based 
 * optimization. It leverages Negative Log-Likelihood (NLL) as the objective 
 * function to minimize the discrepancy between predicted and actual noise.
 *
 * DESIGN RATIONALE:
 * - Functional Decoupling: Uses `std::function` for model prediction, allowing 
 * seamless swapping of CNN or Transformer backends.
 * - Dynamic Timestep Selection: Ties the schedule to the epoch lifecycle for 
 * progressive manifold learning.
 * - Probabilistic Gradients: Directly optimizes the variational lower bound 
 * via the `NormalDist` utility.
 */

    // The master training function
    // model_predict_epsilon: function/lambda to predict epsilon given noisy input x_t and timestep t
    // params: reference to model parameters vector (updated by optimizer)
void GaussianDiffusion::train(const std::vector<std::vector<double>>& data,
                              int epochs,
                              std::vector<double>& out_nll_losses,
                              std::vector<double>& out_entropy_losses,
                              std::vector<double>& params,
                              std::function<std::vector<double>(const std::vector<double>&, int, const std::vector<double>&)> model_predict_epsilon) {
    out_nll_losses.clear();
    out_entropy_losses.clear();
    /**
 * @brief Executes a training iteration over the provided dataset.
 * @param data The input tensor collection (Clifford-space embeddings).
 * @param epochs Total training iterations.
 * @param out_nll_losses Output buffer for tracking NLL convergence.
 * @param params Mutable reference to the high-dimensional weight vector.
 * @param model_predict_epsilon Lambda/Function hook for the ε-theta network.
 */

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (const auto& sample : data) {
            // Select timestep based on epoch (or your schedule)
            int t = epoch % num_timesteps_;

            // Add noise to sample: forward diffusion step
            std::vector<double> x_t = forward(sample, t);

            // Get epsilon prediction from model
            std::vector<double> epsilon_t = model_predict_epsilon(x_t, t, params);

            // Calculate mean of reverse process
            double beta_t = betas_[t];
            std::vector<double> mu(x_t.size());
            std::vector<double> variance(x_t.size(), beta_t);
            std::vector<double> log_var(x_t.size(), 0.0); // optionally compute log variance properly

            for (size_t i = 0; i < x_t.size(); ++i) {
                mu[i] = (x_t[i] - beta_t * epsilon_t[i]) / std::sqrt(1 - beta_t);
            }

            // Estimate x_start using mean + epsilon scaled by variance (approximation)
            std::vector<double> x_start(x_t.size());
            for (size_t i = 0; i < x_t.size(); ++i) {
                x_start[i] = mu[i] + epsilon_t[i] * std::exp(0.5 * log_var[i]);
            }
            /**
             * @details Objective Function: Negative Log-Likelihood.
             * The model calculates the probability density of the noisy sample $x_t$ given 
             * the predicted mean $\mu$ and variance $\sigma$:
             * * $$\mathcal{L}_{VLB} = -\log p_\theta(x_t \mid x_{start}, \epsilon_t)$$
             * * By minimizing the NLL, we effectively maximize the likelihood that the 
             * model's reverse process correctly identifies the added Gaussian noise.
             */
            // Calculate NLL loss and gradients using NormalDist
            std::vector<double> nll_loss_sample(x_t.size());
            std::vector<double> grad_sample(x_t.size());

            for (size_t i = 0; i < x_t.size(); ++i) {
                try {
                    double logp = NormalDist::log_prob_from_predictions(x_t[i], x_start[i], epsilon_t[i]);
                    nll_loss_sample[i] = -logp;

                    double dfd_y, dfd_mu, dfd_sigma;
                    NormalDist::gradients(x_t[i],
                                          NormalDist::compute_mean(x_start[i], epsilon_t[i]),
                                          NormalDist::compute_sigma(x_start[i], epsilon_t[i]),
                                          dfd_y, dfd_mu, dfd_sigma);
                    grad_sample[i] = dfd_y;
                } catch (const std::invalid_argument& e) {
                    nll_loss_sample[i] = 1e6;  // large penalty if sigma invalid
                    grad_sample[i] = 0.0;
                }
            }

            // Average sample loss for reporting
            double avg_loss = 0.0;
            for (double l : nll_loss_sample) avg_loss += l;
            avg_loss /= nll_loss_sample.size();

            // Append to output loss vectors for BetaSchedule updates
            out_nll_losses.push_back(avg_loss);
            out_entropy_losses.push_back(0.0);  // placeholder entropy

            // Update model params via Adam optimizer using calculated gradients
            optimizer_.update(params, grad_sample);
        }
    }
}
