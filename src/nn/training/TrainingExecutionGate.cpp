#include "latent/nn/training/TrainingExecutionGate.hpp"

#include "latent/nn/training/ToyImageReconstructionTask.hpp"

#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace latent::nn::training {

TrainingExecutionGate::TrainingExecutionGate(
    const TrainingPolicy& requested,
    const ImageDataset& dataset,
    ImageTrainingTask& task,
    const std::string& train_split,
    const std::string& validation_split,
    const std::string& test_split)
    : policy_(requested),
      dataset_manifest_(dataset.manifest_id()),
      task_id_(task.task_id())
{
    // Preserve NN.1.07A's schema-v1 contract: validate()/serialize()
    // remain disabled-by-default and must reject enabled execution.
    // The runtime gate explicitly authorises only the supported E.04
    // image reconstruction fixture after every existing policy check.
    if (!requested.execution_enabled) {
        throw std::invalid_argument(
            "Image training execution must be explicitly enabled");
    }
    TrainingPolicy validated_configuration = requested;
    validated_configuration.execution_enabled = false;
    validated_configuration.validate();

    if (policy_.objective != Objective::ReconstructionMSE) {
        throw std::invalid_argument(
            "E.04 runtime supports only image reconstruction MSE");
    }
    if (policy_.dataset_manifest != dataset_manifest_) {
        throw std::invalid_argument(
            "Image dataset content fingerprint does not match policy");
    }
    // The in-memory E.01 dataset exposes these three named splits.
    // Require matching canonical role labels rather than just accepting
    // different strings which cannot be resolved to dataset contents.
    if (train_split != "train" || validation_split != "validation" ||
        test_split != "test" ||
        policy_.train_split != train_split ||
        policy_.validation_split != validation_split ||
        policy_.test_split != test_split) {
        throw std::invalid_argument(
            "Training policy split identifiers do not match image dataset roles");
    }
    if (dataset.split_size(ImageSplit::Train) == 0 ||
        dataset.split_size(ImageSplit::Validation) == 0 ||
        dataset.split_size(ImageSplit::Test) == 0) {
        throw std::invalid_argument("Missing required image dataset split");
    }

    // NN.1.07E.02 is the only implemented training task. A task
    // implementing the same task_id() but a different execution
    // contract must not enter this preliminary gate.
    if (dynamic_cast<ToyImageReconstructionTask*>(&task) == nullptr) {
        throw std::invalid_argument(
            "E.04 runtime supports only ToyImageReconstructionTask");
    }
    const auto& shape = dataset.image_shape();
    const std::string expected_task =
        "toy_image_reconstruction_linear_mse_v1:" +
        std::to_string(shape[0]) + "x" +
        std::to_string(shape[1]) + "x" +
        std::to_string(shape[2]);
    if (task_id_ != expected_task) {
        throw std::invalid_argument(
            "Reconstruction task shape does not match image dataset");
    }

    std::unordered_set<latent::nn::Parameter*> seen;
    const auto parameters = task.parameters();
    if (parameters.empty()) {
        throw std::invalid_argument("Image training task has no parameters");
    }
    for (auto* parameter : parameters) {
        if (!parameter || !seen.insert(parameter).second) {
            throw std::invalid_argument(
                "Image training parameters must be unique and non-null");
        }
        if (!parameter->values().same_shape(parameter->gradients()) ||
            !parameter->values().all_finite() ||
            !parameter->gradients().all_finite()) {
            throw std::invalid_argument(
                "Invalid image training parameter tensors");
        }
    }

    // E.04 deliberately does not call task.train_microbatch(), mutate
    // gradients, create Adam state, or start the training lifecycle.
}

const TrainingPolicy& TrainingExecutionGate::policy() const noexcept
{
    return policy_;
}

const std::string& TrainingExecutionGate::dataset_manifest() const noexcept
{
    return dataset_manifest_;
}

const std::string& TrainingExecutionGate::task_id() const noexcept
{
    return task_id_;
}

} // namespace latent::nn::training
