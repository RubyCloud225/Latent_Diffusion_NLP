#pragma once

#include "latent/nn/training/ImageDataset.hpp"
#include "latent/nn/training/ImageTrainingTask.hpp"

#include <cstddef>

namespace latent::nn::training {

/**
 * @brief Measured result of one sample-weighted gradient window.
 *
 * @details A window updates gradients only. The caller is responsible
 * for clipping, Adam stepping and clearing gradients at the next
 * successful window boundary (NN.1.07E.07).
 */
struct AccumulationReport {
    double mean_loss = 0.0;
    std::size_t microbatches = 0;
    std::size_t samples = 0;
    bool partial_window = false;
    bool end_of_epoch = false;
};

/**
 * @brief CPU FP64 sample-weighted image gradient accumulator.
 *
 * @details For microbatch j containing n_j images, calls
 * train_microbatch(batch, n_j), then divides accumulated gradients
 * by the total number of images N. The result is the gradient of
 * the sample-weighted mean objective, not an unweighted mean of
 * microbatch gradients.
 *
 * Requires zero gradients on entry: existing accumulated gradients
 * must first be consumed by a later optimiser step and zeroed.
 * On an exception, restores original parameter tensors and the
 * deterministic source cursor and clears task transient state.
 *
 * Does not call the optimiser, alter its moments, increment steps,
 * or advance epochs. Does not support task-owned external RNG or
 * other undeclared side effects; full transaction integration is
 * deferred to NN.1.07E.07.
 *
 * @warning The task and dataset source must outlive the accumulator.
 * @warning Not thread-safe.
 */
class GradientAccumulator final {
public:
    /** @brief Bind an image task/source with positive window length. */
    GradientAccumulator(
        ImageTrainingTask& task,
        ImageBatchSource& source,
        std::size_t accumulation_steps);

    /**
     * @brief Prepare and commit gradients for one accumulation window.
     * @return Sample-weighted loss, sample count and completion flags.
     * @throws std::invalid_argument Invalid task/registration/gradients.
     * @throws std::overflow_error Unrepresentable counts, loss or gradient.
     */
    [[nodiscard]] AccumulationReport accumulate_window();

private:
    ImageTrainingTask& task_;
    ImageBatchSource& source_;
    std::size_t accumulation_steps_;
};

} // namespace latent::nn::training
