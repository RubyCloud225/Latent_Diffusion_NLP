#include "NormalDist.hpp"
#include <cmath>
#include <stdexcept>

namespace NormalDist {
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
    void gradients(double y, double mean, double sigma, double& dfd_y, double& dfd_mu, double& dfd_sigma) {
        dfd_y = grad_wrt_y(y, mean, sigma);
        dfd_mu = -dfd_y;
        dfd_sigma = grad_wrt_sigma(y, mean, sigma);
    }
}