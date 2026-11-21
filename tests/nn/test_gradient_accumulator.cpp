#include "latent/nn/training/GradientAccumulator.hpp"
#include "latent/nn/training/ToyImageReconstructionTask.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using latent::core::Tensor;
using latent::nn::Parameter;
using latent::nn::training::AccumulationReport;
using latent::nn::training::GradientAccumulator;
using latent::nn::training::ImageBatch;
using latent::nn::training::ImageDataset;
using latent::nn::training::ImageSample;
using latent::nn::training::ImageSplit;
using latent::nn::training::ImageTrainingTask;
using latent::nn::training::ToyImageReconstructionTask;

namespace {

void require(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}

void near(double actual, double expected, double tol = 1e-12)
{
    require(std::isfinite(actual) &&
            std::abs(actual - expected) <= tol,
            "Accumulation numerical mismatch");
}

template<class Exception, class Function>
void rejects(Function function)
{
    bool caught = false;
    try { function(); }
    catch (const Exception&) { caught = true; }
    require(caught, "Expected accumulation rejection missing");
}

ImageSample sample(const std::string& id, ImageSplit split, double value)
{
    Tensor image({1, 1, 1});
    image[0] = value;
    return {id, split, std::move(image)};
}

ImageDataset image_dataset(const std::vector<double>& train)
{
    std::vector<ImageSample> samples;
    for (std::size_t i = 0; i < train.size(); ++i) {
        samples.push_back(
            sample("train_" + std::to_string(i), ImageSplit::Train,
                   train[i]));
    }
    samples.push_back(sample("validation", ImageSplit::Validation, 0.3));
    samples.push_back(sample("test", ImageSplit::Test, 0.4));
    return ImageDataset(std::move(samples));
}

void test_full_128_image_window()
{
    ImageDataset dataset = image_dataset(std::vector<double>(128, 0.5));
    auto source = dataset.source(ImageSplit::Train, 16, 42);
    ToyImageReconstructionTask task(1, 1, 1);
    GradientAccumulator accumulator(task, source, 8);
    const auto report = accumulator.accumulate_window();

    require(report.samples == 128 && report.microbatches == 8,
            "Expected eight microbatches / 128 images");
    require(!report.partial_window && !report.end_of_epoch,
            "Full window marked partial");
    near(report.mean_loss, 0.25);

    const auto parameters = task.parameters();
    near(parameters[0]->gradients()[0], -0.5);
    near(parameters[1]->gradients()[0], -1.0);
    near(parameters[0]->values()[0], 0.0);
    near(parameters[1]->values()[0], 0.0);
    require(source.position() == 128, "Wrong full-window cursor");
    // E.03 never steps Adam or changes model parameter values.
}

void test_partial_window_sample_weights()
{
    ImageDataset dataset = image_dataset({0.1, 0.2, 0.3, 0.4, 0.5});
    auto source = dataset.source(ImageSplit::Train, 2, 42);
    ToyImageReconstructionTask task(1, 1, 1);
    GradientAccumulator accumulator(task, source, 8);

    const auto report = accumulator.accumulate_window();
    require(report.samples == 5 && report.microbatches == 3,
            "Partial accumulation counts incorrect");
    require(report.partial_window && report.end_of_epoch,
            "Missing partial/end-of-epoch flags");
    near(report.mean_loss, 0.11);
    auto params = task.parameters();
    near(params[0]->gradients()[0], -0.22);
    near(params[1]->gradients()[0], -0.6);

    params[0]->zero_grad();
    params[1]->zero_grad();
    const auto empty = accumulator.accumulate_window();
    require(empty.samples == 0 && empty.microbatches == 0 &&
            empty.end_of_epoch, "End of epoch should be empty");
    near(params[0]->gradients()[0], 0.0);
    near(params[1]->gradients()[0], 0.0);
}

void test_final_short_microbatch_even_at_full_window_count()
{
    ImageDataset dataset = image_dataset({0.5, 0.5, 0.5});
    auto source = dataset.source(ImageSplit::Train, 2, 42);
    ToyImageReconstructionTask task(1, 1, 1);
    GradientAccumulator accumulator(task, source, 2);
    const auto report = accumulator.accumulate_window();
    require(report.microbatches == 2 && report.samples == 3 &&
            report.partial_window, "Final short microbatch not reported");
    near(report.mean_loss, 0.25);
    near(task.parameters()[0]->gradients()[0], -0.5);
}

void test_reproducible_window()
{
    ImageDataset dataset = image_dataset({0.1, 0.2, 0.3, 0.4, 0.5});
    auto first_source = dataset.source(ImageSplit::Train, 2, 17);
    auto second_source = dataset.source(ImageSplit::Train, 2, 17);
    ToyImageReconstructionTask first_task(1, 1, 1);
    ToyImageReconstructionTask second_task(1, 1, 1);
    GradientAccumulator first(first_task, first_source, 2);
    GradientAccumulator second(second_task, second_source, 2);

    const auto a = first.accumulate_window();
    const auto b = second.accumulate_window();
    near(a.mean_loss, b.mean_loss, 0.0);
    require(a.samples == b.samples &&
            first_source.checkpoint_cursor() ==
                second_source.checkpoint_cursor(),
            "Seeded accumulation cursor changed");
    for (std::size_t i = 0; i < 2; ++i) {
        near(first_task.parameters()[i]->gradients()[0],
             second_task.parameters()[i]->gradients()[0], 0.0);
    }
}

class InjectedFailureTask final : public ImageTrainingTask {
public:
    enum class Failure { Throw, InvalidLoss, NonFiniteGradient, Mutation };
    explicit InjectedFailureTask(Failure failure)
        : failure_(failure), parameter_({1}) {}

    std::string task_id() const override { return "failure-fixture"; }
    std::vector<Parameter*> parameters() override { return {&parameter_}; }

    double train_microbatch(const ImageBatch&, double) override
    {
        ++calls_;
        parameter_.gradients()[0] += 4.0;
        if (calls_ == 2) {
            if (failure_ == Failure::Throw) {
                throw std::runtime_error("Injected backward failure");
            }
            if (failure_ == Failure::InvalidLoss) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            if (failure_ == Failure::NonFiniteGradient) {
                parameter_.gradients()[0] =
                    std::numeric_limits<double>::infinity();
            }
            if (failure_ == Failure::Mutation) {
                parameter_.values()[0] = 5.0;
            }
        }
        return 0.25;
    }

    double validation_loss(const ImageBatch&) const override { return 0.25; }
    void reset_transient_state() noexcept override { reset_ = true; }
    Parameter& parameter() noexcept { return parameter_; }
    bool reset_called() const noexcept { return reset_; }

private:
    Failure failure_;
    Parameter parameter_;
    std::size_t calls_ = 0;
    bool reset_ = false;
};

void test_rollback_for(InjectedFailureTask::Failure mode)
{
    ImageDataset dataset = image_dataset({0.1, 0.2, 0.3});
    auto source = dataset.source(ImageSplit::Train, 1, 19);
    InjectedFailureTask task(mode);
    GradientAccumulator accumulator(task, source, 3);
    const auto initial_cursor = source.checkpoint_cursor();

    bool rejected = false;
    try {
        (void)accumulator.accumulate_window();
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "Injected failure not rejected");
    require(source.checkpoint_cursor() == initial_cursor,
            "Failed window changed dataset cursor");
    near(task.parameter().gradients()[0], 0.0);
    near(task.parameter().values()[0], 0.0);
    require(task.reset_called(), "Failure did not reset transient state");
}

void test_invalid_boundary()
{
    ImageDataset dataset = image_dataset({0.2, 0.3});
    auto source = dataset.source(ImageSplit::Train, 1, 19);
    ToyImageReconstructionTask task(1, 1, 1);
    rejects<std::invalid_argument>([&] {
        GradientAccumulator invalid(task, source, 0);
    });

    task.parameters()[0]->gradients()[0] = 2.0;
    const auto cursor = source.checkpoint_cursor();
    GradientAccumulator accumulator(task, source, 2);
    rejects<std::invalid_argument>([&] {
        (void)accumulator.accumulate_window();
    });
    require(source.checkpoint_cursor() == cursor,
            "Invalid start advanced source");
    near(task.parameters()[0]->gradients()[0], 2.0);
}

} // namespace

int main()
{
    try {
        test_full_128_image_window();
        test_partial_window_sample_weights();
        test_final_short_microbatch_even_at_full_window_count();
        test_reproducible_window();
        test_rollback_for(InjectedFailureTask::Failure::Throw);
        test_rollback_for(InjectedFailureTask::Failure::InvalidLoss);
        test_rollback_for(InjectedFailureTask::Failure::NonFiniteGradient);
        test_rollback_for(InjectedFailureTask::Failure::Mutation);
        test_invalid_boundary();
        std::cout << "[PASS] NN.1.07E.03 gradient accumulator\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] NN.1.07E.03: " << error.what() << '\n';
        return 1;
    }
}
