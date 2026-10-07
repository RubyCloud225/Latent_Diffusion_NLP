#ifndef NORMALDIST_HPP
#define NORMALDIST_HPP
#include <cmath>

namespace NormalDist {
    double log_prob(double y, double mean, double sigma);
    double compute_mean(double x_start_pred, double eps_pred);
    double compute_sigma(double x_start_pred, double eps_pred);
    double log_prob_from_predictions(double y, double x_start_pred, double eps_pred);
    double grad_wrt_y(double y, double mean, double sigma);
    double grad_wrt_sigma(double y, double mean, double sigma);
    void gradients(double y, double mean, double sigma, double& dfd_y, double& dfd_mu, double& dfd_sigma);
}

#endif // NORMALDIST_HPP