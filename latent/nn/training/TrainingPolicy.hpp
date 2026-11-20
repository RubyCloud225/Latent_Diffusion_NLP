#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace latent::nn::training {

/** @brief Task-specific objective, independently validated by the trainer. */
enum class Objective {
    ReconstructionMSE,
    EpsilonMSE
};

/** @brief Supported gradient-control policy choices. */
enum class GradientMode {
    None,
    EuclideanGlobal,
    MetricRadial
};

/**
 * @brief Versioned, serialisable FP64 CPU training policy.
 *
 * @details Holds configuration only; creates no optimiser, dataset,
 * model, checkpoint or gradient controller. A runtime must copy and
 * validate a policy before beginning execution.
 *
 * @note Split names are distinct identifiers, not proof that samples
 * are disjoint. Dataset-manifest verification is a separate task.
 * @warning Training execution is deliberately prohibited in NN.1.07A.
 */
struct TrainingPolicy {
    static constexpr std::uint32_t schema_version = 1;

    Objective objective = Objective::ReconstructionMSE;
    std::string dataset_manifest;
    std::string train_split;
    std::string validation_split;
    std::string test_split;
    std::string code_revision;

    std::uint64_t seed = 42;
    std::size_t batch_size = 16;
    std::size_t accumulation_steps = 8;
    std::size_t max_optimiser_steps = 1000;

    double learning_rate = 0.001;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;

    GradientMode gradient_mode = GradientMode::None;
    double max_gradient_norm = 1.0;
    std::string metric_id;

    std::size_t validation_interval = 100;
    std::size_t checkpoint_interval = 100;
    std::size_t max_consecutive_failures = 1;
    bool record_gradient_norms = true;
    bool save_optimizer_state = true;
    bool save_rng_state = true;

    bool execution_enabled = false;

    /**
     * @brief Reject invalid, unsupported or inconsistent policy values.
     * @throws std::invalid_argument On invalid configuration.
     * @throws std::overflow_error On effective-batch overflow.
     */
    void validate() const;

    /**
     * @brief Multiply batch size by accumulation steps, with overflow check.
     * @return Nominal samples per full accumulation window.
     * @throws std::overflow_error On multiplication overflow.
     */
    std::size_t effective_batch_size() const;

    /**
     * @brief Produce canonical, deterministic policy-v1 UTF-8 text.
     * @return Ordered key/value representation, round-trip-safe for FP64.
     * @throws std::invalid_argument For an invalid policy.
     */
    std::string serialize() const;

    /**
     * @brief Read canonical policy-v1 text and validate it.
     * @param text Complete policy record.
     * @return Validated policy value.
     * @throws std::invalid_argument On missing, unknown, duplicate,
     *         malformed or unsupported fields.
     * @throws std::overflow_error For numeric size overflow.
     */
    static TrainingPolicy parse(const std::string& text);
};

} // namespace latent::nn::training
