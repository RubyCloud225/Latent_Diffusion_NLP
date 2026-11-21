
#pragma once

#include "latent/nn/Parameter.hpp"

#include <cstddef>
#include <vector>

namespace latent::nn::training {

/**
 * @brief Diagnostics for one gradient measurement
 * or Euclidean constraint operation.
 */
struct GradientReport {
    double norm_before = 0.0;
    double norm_after = 0.0;
    double applied_scale = 1.0;

    bool clipped = false;

    std::size_t parameter_blocks = 0;
    std::size_t parameter_count = 0;
};

/**
 * @brief FP64 global Euclidean gradient controller.
 *
 * Measures all registered Parameter gradients as one
 * concatenated vector.
 *
 * @details Uses a stable scaled sum-of-squares norm.
 * Failed validation or staging must leave every
 * registered gradient unchanged.
 *
 * @note Does not update parameters or Adam moments.
 * @warning Not thread-safe during gradient mutation.
 */
class GradientController final {
public:
    /**
     * @brief Construct the controller.
     * @param max_norm Positive, finite L2 threshold.
     */
    explicit GradientController(double max_norm);

    /**
     * @brief Measure without modifying gradients.
     * @param parameters Unique, non-null pointers.
     * @return Read-only global norm and diagnostics.
     */
    [[nodiscard]]
    GradientReport measure(
        const std::vector<Parameter*>& parameters
    ) const;

    /**
     * @brief Optionally clip accumulated gradients.
     *
     * @details Computes one global scaling factor.
     * All scaled gradients are staged and validated
     * before any gradient Tensor is replaced.
     */
    GradientReport apply(
        const std::vector<Parameter*>& parameters
    ) const;

private:
    double max_norm_;
};

} // namespace latent::nn::training
