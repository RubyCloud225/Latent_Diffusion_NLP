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


/**
 * @brief Sigmoid Linear Unit (SiLU, also called Swish).
 *
 * @details Computes x * sigmoid(x) elementwise with a numerically stable
 * sigmoid branch for positive and negative FP64 inputs. Backpropagation uses
 * sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)).
 *
 * SiLU has no trainable parameters and preserves Tensor shape and rank.
 * The most recent successful forward input is cached for backward().
 *
 * @warning Do not train one instance concurrently from multiple threads.
 */
class SiLU final : public Layer {
public:
    /**
     * @brief Apply SiLU elementwise without modifying the input.
     * @param input Finite FP64 Tensor.
     * @return Newly allocated Tensor of identical shape.
     * @throws std::invalid_argument Non-finite input.
     * @throws std::overflow_error Non-finite computed output.
     */
    latent::core::Tensor forward(
        const latent::core::Tensor& input
    ) override;

    /**
     * @brief Backpropagate through the latest successful forward input.
     * @param grad_output Finite upstream gradient with matching shape.
     * @return Input gradient with identical shape.
     * @throws std::logic_error If forward was not called.
     * @throws std::invalid_argument Gradient shape/finiteness error.
     * @throws std::overflow_error Non-finite computed gradient.
     */
    latent::core::Tensor backward(
        const latent::core::Tensor& grad_output
    ) override;

    /** @return Empty parameter list because SiLU has no trainable weights. */
    std::vector<Parameter*> parameters() override;

private:
    latent::core::Tensor cached_input_{{1}};
    bool has_forward_ = false;

    /**
     * @brief Evaluate sigmoid with overflow-safe exponentials.
     * @param x Finite FP64 input.
     * @return Value within [0, 1].
     */
    static double sigmoid(double x) noexcept;
};

} // namespace latent::nn
