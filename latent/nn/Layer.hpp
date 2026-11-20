
#pragma once

#include "latent/core/Tensor.hpp"
#include "latent/nn/Parameter.hpp"

#include <vector>

namespace latent::nn {

/**
 * @brief Abstract trainable neural-network layer.
 *
 * All implementations use FP64 CPU Tensors.
 *
 * A training iteration typically consists of:
 *
 * 1. zero_grad()
 * 2. forward(input)
 * 3. calculate loss
 * 4. backward(gradient)
 * 5. optimiser update
 *
 * Each layer owns its forward-pass cache and
 * trainable parameters where applicable.
 *
 * The base interface is non-copyable because
 * parameter pointers and forward caches belong
 * to a specific layer instance.
 */
class Layer {
public:
    virtual ~Layer() = default;

    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;

    /**
     * @brief Execute the forward pass.
     *
     * @param input Input Tensor.
     * @return Output Tensor.
     */
    virtual latent::core::Tensor forward(
        const latent::core::Tensor& input
    ) = 0;

    /**
     * @brief Backpropagate the output gradient.
     *
     * Accumulates parameter gradients and returns
     * the gradient with respect to the input.
     *
     * @param grad_output Gradient of loss with
     *        respect to the previous forward output.
     * @return Gradient with respect to layer input.
     *
     * @throws std::logic_error If no valid forward
     *         pass has been performed.
     * @throws std::invalid_argument If gradient
     *         shape is incompatible.
     */
    virtual latent::core::Tensor backward(
        const latent::core::Tensor& grad_output
    ) = 0;

    /**
     * @brief Return non-owning parameter pointers.
     *
     * The pointers remain valid while the layer
     * and its parameter storage remain alive.
     */
    virtual std::vector<Parameter*> parameters() = 0;

    /** @brief Clear all accumulated parameter gradients. */
    virtual void zero_grad() {
        for (Parameter* parameter : parameters()) {
            parameter->zero_grad();
        }
    }

protected:
    Layer() = default;
};

} // namespace latent::nn
