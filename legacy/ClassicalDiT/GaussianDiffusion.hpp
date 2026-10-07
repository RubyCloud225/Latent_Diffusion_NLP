#ifndef GAUSSIANDIFFUSION_HPP
#define GAUSSIANDIFFUSION_HPP

#include <vector>
#include <cmath>
#include <random>
#include <iostream>

class AdamOptimizer {
    public: 
    AdamOptimizer(double learning_rate, double beta1, double beta2, double epsilon);
    void update(std::vector<double>& params, std::vector<double>& gradients);
    private:
    double learning_rate_;
    double beta1_;
    double beta2_;
    double epsilon_;
    std::vector<double> m_;
    std::vector<double> v_;
    int t_;
};

class GaussianDiffusion {
    public:
    GaussianDiffusion(int num_timesteps, double beta_start, double beta_end);
    // Forward Process
    std::vector<double> forward(const std::vector<double>& x_prev, int t);
    // Reverse Process
    std::vector<double> reverse(const std::vector<double>& x_t, int t);
    //Train with adam optimizer
     void train(const std::vector<std::vector<double>>& data,
               int epochs,
               std::vector<double>& out_nll_losses,
               std::vector<double>& out_entropy_losses,
               std::vector<double>& params,
               std::function<std::vector<double>(const std::vector<double>&, int, const std::vector<double>&)> model_predict_epsilon);
    private:
    int num_timesteps_;
    double beta_start_;
    double beta_end_;
    std::vector<double> betas_;
    AdamOptimizer optimizer_;
    // Activation Function ( x ): [ \text{sigmoid}(x) = \frac{1}{1 + e^{-x}} ]
    double sigmoid(double x) const;
    // Derivative of Activation Function [ \text{sigmoid}'(x) = \text{sigmoid}(x) \cdot (1 - \text{sigmoid}(x)) \ 3.
    double sigmoid_derivative(double x) const;
};

#endif // GAUSSIANDIFFUSION_HPP