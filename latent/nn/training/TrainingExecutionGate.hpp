#pragma once

#include "latent/nn/training/ImageDataset.hpp"
#include "latent/nn/training/ImageTrainingTask.hpp"
#include "latent/nn/training/TrainingPolicy.hpp"

#include <string>

namespace latent::nn::training {

/**
 * @brief Explicitly validated image reconstruction execution configuration.
 *
 * @details E.04 is an authorization / compatibility gate only: it performs
 * no gradient calculations or optimiser steps, and cannot enable training
 * by itself. Successful construction validates the existing disabled-by-
 * default policy-v1 configuration against the actual image dataset and
 * the attached reconstruction task. The full session owns this value later.
 *
 * The policy-v1 canonical parse/serialize contract remains unchanged:
 * validate()/serialize()/parse() continue to reject execution_enabled=true.
 * E.08 will define the versioned canonical runtime-checkpoint policy
 * representation before any executable checkpoint support is allowed.
 *
 * @warning The dataset and task must outlive a future session; this
 * gate stores only copied configuration and identities, not pointers.
 */
class TrainingExecutionGate final {
public:
    /**
     * @brief Validate a requested execution against actual runtime objects.
     *
     * @param requested Must explicitly set execution_enabled=true.
     * @param dataset Reference image dataset with disjoint split IDs.
     * @param task Image objective adapter.
     * @param train_split Expected policy identifier for training split.
     * @param validation_split Expected policy identifier for validation.
     * @param test_split Expected policy identifier for test.
     * @throws std::invalid_argument Mismatch, disabled flag, unsupported
     *         objective or gradient mode, or invalid registration.
     * @throws std::overflow_error Invalid nominal effective batch size.
     */
    TrainingExecutionGate(
        const TrainingPolicy& requested,
        const ImageDataset& dataset,
        ImageTrainingTask& task,
        const std::string& train_split,
        const std::string& validation_split,
        const std::string& test_split);

    /** @return Validated enabled policy, owned by this gate. */
    [[nodiscard]] const TrainingPolicy& policy() const noexcept;

    /** @return Validated dataset fingerprint. */
    [[nodiscard]] const std::string& dataset_manifest() const noexcept;

    /** @return Stable task identifier captured at validation. */
    [[nodiscard]] const std::string& task_id() const noexcept;

private:
    TrainingPolicy policy_;
    std::string dataset_manifest_;
    std::string task_id_;
};

} // namespace latent::nn::training
