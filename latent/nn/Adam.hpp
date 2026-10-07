#pragma once

#include <cstddef>
#include <vector>

namespace latent::nn {

class Adam {
public:
    Adam(
        float learning_rate = 1e-3f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float epsilon = 1e-8f
    );

    void step(
        std::vector<float>& parameters,
        const std::vector<float>& gradients
    );

    void reset();

    std::size_t step_count() const noexcept;

private:
    float learning_rate_;
    float beta1_;
    float beta2_;
    float epsilon_;

    std::size_t step_{0};

    std::vector<float> first_moment_;
    std::vector<float> second_moment_;
};

} // namespace latent::nn