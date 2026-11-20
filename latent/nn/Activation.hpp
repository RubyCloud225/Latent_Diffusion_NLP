#pragma once

#include "latent/nn/Layer.hpp"

namespace latent::nn {

/**
 * @brief Rectified Linear Unit activation layer.
 *
 * @details Applies max(0, x) elementwise to an FP64 Tensor, preserving
 * shape and rank. During backpropagation the derivative is zero for
 * x <= 0 and one for x > 0. ReLU has no trainable parameters.
 *
 * @note The last successful forward input is cached for backward().
 * @warning Do not use one instance concurrently from multiple threads.
 */
class ReLU final : public Layer {
public:
    /**
     * @brief Compute max(0, x) elementwise.
     * @param input Finite FP64 input Tensor.
     * @return Newly allocated Tensor with identical shape.
     * @throws std::invalid_argument Non-finite input.
     */
    latent::core::Tensor forward(
        const latent::core::Tensor& input
    ) override;

    /**
     * @brief Backpropagate gradients through ReLU.
     * @param grad_output Finite upstream gradient, matching forward output.
     * @return Input gradient with identical shape.
     * @throws std::logic_error If forward was not called.
     * @throws std::invalid_argument Gradient shape or finiteness failure.
     */
    latent::core::Tensor backward(
        const latent::core::Tensor& grad_output
    ) override;

    /** @return Empty parameter list because ReLU has no trainable weights. */
    std::vector<Parameter*> parameters() override;

private:
    latent::core::Tensor cached_input_{{1}};
    bool has_forward_ = false;
};

} // namespace latent::nn
