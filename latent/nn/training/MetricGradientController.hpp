
#pragma once

#include "latent/nn/Parameter.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace latent::nn::training {

/**
 * @brief Diagnostic result of metric-gradient control.
 */
struct MetricGradientReport {
    double norm_before = 0.0;
    double norm_after = 0.0;
    double applied_scale = 1.0;

    bool clipped = false;

    std::size_t parameter_blocks = 0;
    std::size_t parameter_count = 0;

    std::string metric_id = "equal_block_rms_v1";
};

/**
 * @brief Fixed block-diagonal FP64 gradient controller.
 *
 * @details Each registered Parameter represents one
 * block. Metric block b is:
 *
 *      M_b = I / (B * d_b)
 *
 * The controller performs one global radial scaling,
 * preserving the full gradient direction subject to
 * FP64 rounding. Any weighting or clipping that rounds
 * a nonzero component all the way to zero is rejected
 * before modification; nonzero subnormal rounding is
 * still subject to FP64 precision limits.
 *
 * @note Metric-radial scaling is not generally the
 * Euclidean closest-point projection onto an ellipsoid.
 * This v1 metric is fixed, diagonal and block-weighted;
 * it is not a Fisher or natural-gradient approximation.
 *
 * @warning Parameters must outlive the controller.
 * @warning Not thread-safe.
 */
class MetricGradientController final {
public:
    /**
     * @brief Register stable parameter blocks.
     *
     * @param parameters Unique, non-null parameters.
     * @param max_norm Positive finite metric radius.
     *
     * Captures tensor shapes when constructed.
     */
    MetricGradientController(
        std::vector<Parameter*> parameters,
        double max_norm
    );

    /**
     * @brief Read-only metric-norm measurement.
     *
     * @throws std::invalid_argument Invalid registration,
     *         shape or non-finite gradient.
     * @throws std::overflow_error Unrepresentable norm.
     */
    [[nodiscard]]
    MetricGradientReport measure() const;

    /**
     * @brief Apply metric-radial clipping if required.
     *
     * All replacement gradients must be validated
     * before any registered gradient is changed.
     */
    MetricGradientReport apply();

private:
    std::vector<Parameter*> parameters_;

    std::vector<latent::core::Tensor::Shape>
        registered_shapes_;

    double max_norm_;
};

} // namespace latent::nn::training
