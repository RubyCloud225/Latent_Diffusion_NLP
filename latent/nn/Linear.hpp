
#pragma once

#include "latent/nn/Layer.hpp"

#include <cstddef>
#include <vector>

namespace latent::nn {

/**
 * @brief Trainable FP64 fully connected layer.
 *
 * Computes y = Wx + b.
 *
 * Weight shape: [out_features, in_features].
 * Bias shape:   [out_features].
 *
 * Supported inputs:
 * - [in_features]
 * - [batch, in_features]
 *
 * The final dimension represents input features.
 *
 * @note Parameters are initially zero-initialised.
 *       Trainable initialisation is handled separately.
 *
 * @warning Not safe for concurrent training calls.
 */
class Linear final : public Layer {
public:
    /**
     * @brief Construct a fully connected layer.
     * @param in_features Number of input features.
     * @param out_features Number of output features.
     * @throws std::invalid_argument If either is zero.
     */
    Linear(
        std::size_t in_features,
        std::size_t out_features
    );

    /**
     * @brief Calculate the linear transformation.
     * @param input Rank-1 or rank-2 input Tensor.
     * @return New output Tensor.
     * @throws std::invalid_argument Invalid input shape
     *         or non-finite values.
     * @throws std::overflow_error Non-finite output.
     */
    latent::core::Tensor forward(
        const latent::core::Tensor& input
    ) override;

    /**
     * @brief Backpropagate through the linear layer.
     *
     * Calculates input gradients and accumulates
     * weight and bias gradients.
     *
     * @param grad_output Gradient of the layer output.
     * @return Gradient with respect to layer input.
     * @throws std::logic_error If forward was not called.
     * @throws std::invalid_argument Invalid gradient.
     * @throws std::overflow_error Non-finite calculation.
     */
    latent::core::Tensor backward(
        const latent::core::Tensor& grad_output
    ) override;

    /** @return Non-owning weight and bias pointers. */
    std::vector<Parameter*> parameters() override;

    /** @return Mutable weight parameter. */
    Parameter& weights() noexcept;

    /** @return Mutable bias parameter. */
    Parameter& bias() noexcept;

    /** @return Number of input features. */
    std::size_t in_features() const noexcept;

    /** @return Number of output features. */
    std::size_t out_features() const noexcept;

private:
    std::size_t in_features_;
    std::size_t out_features_;

    Parameter weights_;
    Parameter bias_;

    latent::core::Tensor cached_input_{{1}};
    bool has_forward_ = false;
};

} // namespace latent::nn
