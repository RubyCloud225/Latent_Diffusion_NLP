#include "latent/nn/Activation.hpp"

#include <algorithm>
#include <stdexcept>

namespace latent::nn {

latent::core::Tensor ReLU::forward(
    const latent::core::Tensor& input
)
{
    if (!input.all_finite()) {
        throw std::invalid_argument("ReLU requires finite input");
    }

    latent::core::Tensor output(input.shape());

    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = std::max(0.0, input[i]);
    }

    // Cache only after successful computation; preserve caller-owned input.
    cached_input_ = input;
    has_forward_ = true;
    return output;
}

latent::core::Tensor ReLU::backward(
    const latent::core::Tensor& grad_output
)
{
    if (!has_forward_) {
        throw std::logic_error("ReLU backward requires forward");
    }

    if (!grad_output.same_shape(cached_input_)) {
        throw std::invalid_argument("ReLU gradient shape mismatch");
    }

    if (!grad_output.all_finite()) {
        throw std::invalid_argument("ReLU requires finite gradients");
    }

    latent::core::Tensor grad_input(cached_input_.shape());

    for (std::size_t i = 0; i < grad_input.size(); ++i) {
        grad_input[i] = cached_input_[i] > 0.0
            ? grad_output[i]
            : 0.0;
    }

    return grad_input;
}

std::vector<Parameter*> ReLU::parameters()
{
    return {};
}

} // namespace latent::nn
