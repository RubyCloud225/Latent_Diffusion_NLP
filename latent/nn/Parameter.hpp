
#pragma once

#include "latent/core/Tensor.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace latent::nn {

/**
 * @brief Trainable FP64 neural-network parameter.
 *
 * Owns parameter values and a matching gradient Tensor.
 * Gradients are accumulated during backward propagation
 * and reset before the next optimisation step.
 */
class Parameter final {
public:
    /**
     * @brief Construct a parameter with matching storage.
     * @param shape Tensor dimensions.
     */
    explicit Parameter(latent::core::Tensor::Shape shape)
        : values_(shape),
          gradients_(std::move(shape))
    {
    }

    /** @return Mutable parameter values. */
    latent::core::Tensor& values() noexcept {
        return values_;
    }

    /** @return Read-only parameter values. */
    const latent::core::Tensor& values() const noexcept {
        return values_;
    }

    /** @return Mutable accumulated gradients. */
    latent::core::Tensor& gradients() noexcept {
        return gradients_;
    }

    /** @return Read-only accumulated gradients. */
    const latent::core::Tensor& gradients() const noexcept {
        return gradients_;
    }

    /** @brief Reset accumulated gradients to zero. */
    void zero_grad() {
        gradients_.fill(0.0);
    }

    /**
     * @brief Accumulate a gradient Tensor.
     * @param gradient Gradient with identical shape.
     * @throws std::invalid_argument Shape mismatch
     *         or non-finite input.
     * @throws std::overflow_error Non-finite sum.
     */
    void accumulate_gradient(
        const latent::core::Tensor& gradient
    ) {
        if (!gradients_.same_shape(gradient)) {
            throw std::invalid_argument(
                "Parameter gradient shape mismatch"
            );
        }

        if (!gradient.all_finite() ||
            !gradients_.all_finite()) {
            throw std::invalid_argument(
                "Parameter gradients must be finite"
            );
        }

        // Validate the entire update before mutating
        // any accumulated gradients.
        latent::core::Tensor updated = gradients_;

        for (std::size_t i = 0; i < updated.size(); ++i) {
            updated[i] += gradient[i];
        }

        if (!updated.all_finite()) {
            throw std::overflow_error(
                "Gradient accumulation overflow"
            );
        }

        gradients_ = std::move(updated);
    }

private:
    latent::core::Tensor values_;
    latent::core::Tensor gradients_;
};

} // namespace latent::nn
