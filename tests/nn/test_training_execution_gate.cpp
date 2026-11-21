#include "latent/nn/training/TrainingExecutionGate.hpp"
#include "latent/nn/training/ToyImageReconstructionTask.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using latent::core::Tensor;
using latent::nn::training::GradientMode;
using latent::nn::training::ImageDataset;
using latent::nn::training::ImageSample;
using latent::nn::training::ImageSplit;
using latent::nn::training::Objective;
using latent::nn::training::ToyImageReconstructionTask;
using latent::nn::training::TrainingExecutionGate;
using latent::nn::training::TrainingPolicy;

namespace {

void require(bool success, const char* message)
{
    if (!success) throw std::runtime_error(message);
}

template<typename Exception, typename Function>
void rejects(Function operation)
{
    bool caught = false;
    try { operation(); }
    catch (const Exception&) { caught = true; }
    require(caught, "Expected execution-gate rejection missing");
}

ImageDataset dataset_fixture()
{
    std::vector<ImageSample> samples;
    for (unsigned i = 0; i < 6; ++i) {
        Tensor pixels({1, 1, 2});
        pixels.fill(static_cast<double>(i) / 10.0);
        samples.push_back({"train-" + std::to_string(i),
                           ImageSplit::Train, std::move(pixels)});
    }
    Tensor validation({1, 1, 2});
    validation.fill(0.25);
    samples.push_back({"val", ImageSplit::Validation,
                       std::move(validation)});
    Tensor test({1, 1, 2});
    test.fill(0.75);
    samples.push_back({"test", ImageSplit::Test, std::move(test)});
    return ImageDataset(std::move(samples));
}

TrainingPolicy enabled_policy(const ImageDataset& dataset)
{
    TrainingPolicy p;
    p.objective = Objective::ReconstructionMSE;
    p.dataset_manifest = dataset.manifest_id();
    p.train_split = "train";
    p.validation_split = "validation";
    p.test_split = "test";
    p.code_revision = "fixture-revision";
    p.batch_size = 2;
    p.accumulation_steps = 3;
    p.max_optimiser_steps = 10;
    p.execution_enabled = true;
    return p;
}

void accept(const TrainingPolicy& policy, ImageDataset& dataset,
            ToyImageReconstructionTask& task)
{
    const TrainingExecutionGate gate(
        policy, dataset, task, "train", "validation", "test");
    require(gate.policy().execution_enabled,
            "Runtime gate lost enabled flag");
    require(gate.dataset_manifest() == dataset.manifest_id(),
            "Runtime gate lost dataset identity");
    require(gate.task_id() == task.task_id(),
            "Runtime gate lost task identity");
    require(gate.policy().batch_size == 2,
            "Runtime gate changed policy");
}

void test_explicit_gate_and_schema_v1_compatibility()
{
    auto dataset = dataset_fixture();
    ToyImageReconstructionTask task(1, 1, 2);
    auto p = enabled_policy(dataset);

    // Preserve the old policy-v1 promise: an enabled policy cannot
    // be serialized or parsed through the disabled-only schema.
    rejects<std::invalid_argument>([&] { p.validate(); });
    rejects<std::invalid_argument>([&] { (void)p.serialize(); });

    TrainingPolicy disabled = p;
    disabled.execution_enabled = false;
    const std::string canonical = disabled.serialize();
    require(TrainingPolicy::parse(canonical).serialize() == canonical,
            "Policy-v1 canonical disabled roundtrip changed");

    for (const auto mode : {GradientMode::None,
                            GradientMode::EuclideanGlobal,
                            GradientMode::MetricRadial}) {
        auto configured = p;
        configured.gradient_mode = mode;
        if (mode == GradientMode::MetricRadial) {
            configured.metric_id = "equal_block_rms_v1";
        }
        accept(configured, dataset, task);
    }

    // Authorisation is not an optimiser update or a forward pass.
    for (auto* parameter : task.parameters()) {
        for (std::size_t i = 0; i < parameter->values().size(); ++i) {
            require(parameter->values()[i] == 0.0,
                    "Gate modified parameter values");
            require(parameter->gradients()[i] == 0.0,
                    "Gate modified parameter gradients");
        }
    }
}

void test_disabled_rejected()
{
    auto dataset = dataset_fixture();
    ToyImageReconstructionTask task(1, 1, 2);
    auto p = enabled_policy(dataset);
    p.execution_enabled = false;
    rejects<std::invalid_argument>([&] {
        accept(p, dataset, task);
    });
}

void test_dataset_and_split_identity_rejected()
{
    auto dataset = dataset_fixture();
    ToyImageReconstructionTask task(1, 1, 2);
    auto p = enabled_policy(dataset);

    p.dataset_manifest = "fnv1a64:0000000000000000";
    rejects<std::invalid_argument>([&] {
        accept(p, dataset, task);
    });

    p = enabled_policy(dataset);
    p.train_split = "other";
    rejects<std::invalid_argument>([&] {
        accept(p, dataset, task);
    });

    p = enabled_policy(dataset);
    rejects<std::invalid_argument>([&] {
        TrainingExecutionGate gate(
            p, dataset, task, "validation", "train", "test");
    });
}

void test_unsupported_objective_and_image_shape()
{
    auto dataset = dataset_fixture();
    ToyImageReconstructionTask task(1, 1, 2);
    auto p = enabled_policy(dataset);
    p.objective = Objective::EpsilonMSE;
    rejects<std::invalid_argument>([&] {
        accept(p, dataset, task);
    });

    p = enabled_policy(dataset);
    ToyImageReconstructionTask wrong_shape(1, 2, 1);
    rejects<std::invalid_argument>([&] {
        accept(p, dataset, wrong_shape);
    });

    p = enabled_policy(dataset);
    p.gradient_mode = GradientMode::MetricRadial;
    p.metric_id = "other_metric";
    rejects<std::invalid_argument>([&] {
        accept(p, dataset, task);
    });
}

void test_reject_bad_runtime_values()
{
    auto dataset = dataset_fixture();
    ToyImageReconstructionTask task(1, 1, 2);
    auto p = enabled_policy(dataset);
    p.learning_rate = std::numeric_limits<double>::quiet_NaN();
    rejects<std::invalid_argument>([&] {
        accept(p, dataset, task);
    });

    p = enabled_policy(dataset);
    p.code_revision.clear();
    rejects<std::invalid_argument>([&] {
        accept(p, dataset, task);
    });

    p = enabled_policy(dataset);
    p.accumulation_steps = 0;
    rejects<std::invalid_argument>([&] {
        accept(p, dataset, task);
    });

    p = enabled_policy(dataset);
    task.parameters()[0]->gradients()[0] =
        std::numeric_limits<double>::infinity();
    rejects<std::invalid_argument>([&] {
        accept(p, dataset, task);
    });
}

} // namespace

int main()
{
    try {
        test_explicit_gate_and_schema_v1_compatibility();
        test_disabled_rejected();
        test_dataset_and_split_identity_rejected();
        test_unsupported_objective_and_image_shape();
        test_reject_bad_runtime_values();
        std::cout << "[PASS] NN.1.07E.04 training execution gate\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] NN.1.07E.04: " << error.what() << '\n';
        return 1;
    }
}
