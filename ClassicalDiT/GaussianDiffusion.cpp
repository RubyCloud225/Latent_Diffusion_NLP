#include "GaussianDiffusion.hpp"
#include "NN/EpsilonPredictor.hpp"
#include "NormalDist.hpp"
//#include <omp.h>
#include <stdexcept> // for std::invalid_argument

// Adam Optimizer Constructor
AdamOptimizer::AdamOptimizer(double learning_rate, double beta1, double beta2, double epsilon) : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0) {
    // Initialize first and second moment vectors
}

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