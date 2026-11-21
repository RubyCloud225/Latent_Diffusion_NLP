#pragma once

#include "latent/nn/Parameter.hpp"
#include "latent/nn/training/ImageBatch.hpp"

#include <string>
#include <vector>

namespace latent::nn::training {

/**
 * @brief Non-owning training-session interface for image objectives.
 *
 * @details The caller owns the task, its parameters, and any associated
 * model. Return the unscaled mean loss. For each microbatch, accumulate
 * loss_scale * d(mean_loss)/d(parameters) into existing gradients.
 * A failing microbatch must not partially modify those gradients.
 *
 * @warning Not thread-safe during training or parameter mutation.
 */
class ImageTrainingTask {
public:
    virtual ~ImageTrainingTask() = default;

    /** @return Stable identifier for task/checkpoint compatibility. */
    [[nodiscard]] virtual std::string task_id() const = 0;

    /** @return Registered trainable parameters in stable order. */
    [[nodiscard]] virtual std::vector<latent::nn::Parameter*>
    parameters() = 0;

    /**
     * @brief Train one batch, accumulating scaled mean-loss gradients.
     * @param batch Valid FP64 image batch.
     * @param loss_scale Finite, non-negative accumulation scale.
     * @return Unscaled mean objective for the batch.
     */
    virtual double train_microbatch(
        const ImageBatch& batch, double loss_scale) = 0;

    /**
     * @brief Held-out batch mean loss without changing model or gradients.
     */
    [[nodiscard]] virtual double validation_loss(
        const ImageBatch& batch) const = 0;

    /** @brief Invalidate task-level transient training state. */
    virtual void reset_transient_state() noexcept = 0;
};

} // namespace latent::nn::training
