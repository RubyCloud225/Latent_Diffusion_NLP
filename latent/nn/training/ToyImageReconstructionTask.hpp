#pragma once

#include "latent/nn/Linear.hpp"
#include "latent/nn/training/ImageTrainingTask.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace latent::nn::training {

/**
 * @brief Minimal learnable FP64 image-to-image reconstruction fixture.
 *
 * @details Flattens NCHW images to [N, C*H*W], runs one Linear layer,
 * and minimises mean squared error across all image elements.
 * This fixture is NOT an autoencoder, diffusion denoiser or image
 * generator. Its purpose is to validate objective/backward integration.
 *
 * This class is not thread-safe. It owns its Linear layer and parameters.
 */
class ToyImageReconstructionTask final : public ImageTrainingTask {
public:
    /**
     * @brief Construct a shape-specific reconstruction fixture.
     * @param channels Image channels (1 or 3).
     * @param height Positive image height.
     * @param width Positive image width.
     * @throws std::overflow_error On flattened-size overflow.
     */
    ToyImageReconstructionTask(
        std::size_t channels,
        std::size_t height,
        std::size_t width);

    [[nodiscard]] std::string task_id() const override;
    [[nodiscard]] std::vector<latent::nn::Parameter*>
    parameters() override;

    double train_microbatch(
        const ImageBatch& batch, double loss_scale) override;

    [[nodiscard]] double validation_loss(
        const ImageBatch& batch) const override;

    void reset_transient_state() noexcept override;

    /** @return Owned reference layer for fixture inspection/setup. */
    [[nodiscard]] latent::nn::Linear& layer() noexcept;

private:
    void require_batch(const ImageBatch& batch) const;
    [[nodiscard]] latent::core::Tensor flatten(
        const ImageBatch& batch) const;

    std::size_t channels_;
    std::size_t height_;
    std::size_t width_;
    std::size_t features_;
    latent::nn::Linear layer_;
    bool transient_valid_ = false;
};

} // namespace latent::nn::training
