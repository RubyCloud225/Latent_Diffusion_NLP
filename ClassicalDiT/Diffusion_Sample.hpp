#ifndef DIFFUSION_SAMPLE_HPP
#define DIFFUSION_SAMPLE_HPP

#include <vector>
#include <unordered_map>
#include <functional>
#include <random>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iterator>

class DiffusionModel;

class DiffusionSample {
    public:
    DiffusionSample(DiffusionModel& model, const std::vector<double>& noise_schedule);
    // Generate a batch of samples
    std::vector<std::vector<double> > p_sample(
        const std::vector<int>& shape,
        bool clip_denoised,
        const std::function<std::vector<double>(const std::vector<double>&)>& denoised_fn,
        const std::unordered_map<std::string, double>& model_kwags,
        const std::string& device
    );
    
    // Generate a single sample
    std::vector<std::unordered_map<std::string, std::vector<double>>> p_sample_loop_progressive(
        const std::vector<int>& shape,
        bool clip_denoised,
        const std::function<std::vector<double>(const std::vector<double>&)>& denoised_fn,
        const std::unordered_map<std::string, double>& model_kwags,
        const std::string& device);
    private:
    DiffusionModel& model_;
    std::vector<double> noise_schedule_;
    std::default_random_engine generator_;
    std::normal_distribution<double> normal_dist_;
public:
    static double clamp(double value, double min_val, double max_val) {
        return std::max(min_val, std::min(value, max_val));
    }
};

#endif // DIFFUSION_SAMPLE_HPP
