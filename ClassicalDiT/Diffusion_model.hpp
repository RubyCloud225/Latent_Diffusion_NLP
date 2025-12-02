#ifndef DIFFUSION_MODEL_HPP
#define DIFFUSION_MODEL_HPP

#include <vector>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <functional>
#include <cmath>

class DiffusionModel {
    public:
    DiffusionModel(int input_size, int output_size);
    // compute P_mean_variance
    void compute_mean_variance(const std::vector<double>& x_t, int t, std::vector<double>& mean, std::vector<double>& variance);
    // Method to sample x_{t-1} from x_t
    std::vector<double> sample(
        const std::vector<double>& x, 
        int t, 
        bool clip_denoised = true, 
        const std::function<std::vector<double>(const std::vector<double>&)>& denoised_fn = nullptr,
        const std::function<std::vector<double>(const std::vector<double>&)>& cond_fn = nullptr,
        const std::unordered_map<std::string, double>& model_kwags = {}
    );
    private:
    int input_size;
    int output_size;
    std::default_random_engine generator;
    std::normal_distribution<double> normal_dist;
    void clamp_vector(std::vector<double>& vec, double min_val, double max_val);
};

#endif // DIFFUSION_MODEL_HPPß