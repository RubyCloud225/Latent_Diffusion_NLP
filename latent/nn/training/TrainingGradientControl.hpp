#pragma once

#include "latent/nn/training/GradientController.hpp"
#include "latent/nn/training/MetricGradientController.hpp"
#include "latent/nn/training/TrainingPolicy.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace latent::nn::training {

/**
 * @brief One measured gradient-control application or read-only probe.
 *
 * @details Norms use the named norm_kind, not necessarily Euclidean:
 * "euclidean_l2" for None and EuclideanGlobal; "equal_block_rms_v1"
 * for MetricRadial. norm_after is independently measured after
 * actual gradient arithmetic, never an assumed clipping threshold.
 */
struct TrainingGradientMetrics {
    GradientMode mode = GradientMode::None;
    std::string norm_kind = "euclidean_l2";
    std::string metric_id;
    double norm_before = 0.0;
    double norm_after = 0.0;
    double applied_scale = 1.0;
    bool clipped = false;
    std::size_t parameter_blocks = 0;
    std::size_t parameter_count = 0;
};

/**
 * @brief Cumulative statistics from successful calls to apply().
 *
 * @note These count controller operations, not optimiser commits.
 * The E.07 session will distinguish operations from committed Adam
 * steps and implement whole-step rollback of prior clipping.
 */
struct GradientControlCounters {
    std::size_t successful_applications = 0;
    std::size_t clipped_applications = 0;
    std::size_t rejected_applications = 0;
};

/**
 * @brief Validated mode dispatch for existing NN.1.07C/D controllers.
 *
 * @details Stores non-owning, stable Parameter pointers and initial
 * tensor shapes. A failed application restores all registered gradients
 * and leaves successful/clipped counters unchanged. This transaction
 * ends at the gradient-controller boundary: it does not protect against
 * a later Adam failure (NN.1.07E.07).
 *
 * @warning The registered Parameters must outlive the control object.
 * @warning Not thread-safe.
 */
class TrainingGradientControl final {
public:
    TrainingGradientControl(
        std::vector<latent::nn::Parameter*> parameters,
        GradientMode mode,
        double max_norm,
        std::string metric_id = {});

    /** @brief Measure without changing gradients or counters. */
    [[nodiscard]] TrainingGradientMetrics measure() const;

    /** @brief Apply configured clipping and report measured post-norm. */
    TrainingGradientMetrics apply();

    /** @return Controller operation counters. */
    [[nodiscard]] GradientControlCounters counters() const noexcept;

private:
    void validate_registration() const;
    void record_rejection() noexcept;

    std::vector<latent::nn::Parameter*> parameters_;
    std::vector<latent::core::Tensor::Shape> shapes_;
    GradientMode mode_;
    double max_norm_;
    std::string metric_id_;
    std::unique_ptr<MetricGradientController> metric_controller_;
    GradientControlCounters counters_;
};

} // namespace latent::nn::training
