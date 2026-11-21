#include "latent/nn/training/ToyImageReconstructionTask.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using latent::core::Tensor;
using latent::nn::training::ImageBatch;
using latent::nn::training::ToyImageReconstructionTask;

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void near(double actual, double expected, double tolerance = 1e-12)
{
    require(std::isfinite(actual) &&
            std::abs(actual - expected) <= tolerance,
            "Image reconstruction numerical mismatch");
}

template<typename Exception, typename F>
void rejects(F operation)
{
    bool caught = false;
    try { operation(); }
    catch (const Exception&) { caught = true; }
    require(caught, "Expected rejection missing");
}

ImageBatch one_pixel(double pixel)
{
    Tensor pixels({1, 1, 1, 1});
    pixels[0] = pixel;
    return ImageBatch(std::move(pixels), {"image-1"});
}

void test_known_answer_and_scaled_backprop()
{
    ToyImageReconstructionTask task(1, 1, 1);
    const auto batch = one_pixel(0.5);

    auto registered = task.parameters();
    require(registered.size() == 2, "Expected weights and bias");
    require(task.task_id() == "toy_image_reconstruction_linear_mse_v1:1x1x1",
            "Unstable task identity");
    // Default Linear weight and bias are zero:
    // MSE = (0 - 0.5)^2 = 0.25.
    // dL/dW = 2 * (-0.5) * (0.5) = -0.5;
    // dL/db = 2 * (-0.5) = -1.
    near(task.train_microbatch(batch, 2.0), 0.25);
    near(registered[0]->gradients()[0], -1.0);
    near(registered[1]->gradients()[0], -2.0);
    near(registered[0]->values()[0], 0.0);
    near(registered[1]->values()[0], 0.0);

    // Backprop must accumulate rather than overwrite.
    near(task.train_microbatch(batch, 1.0), 0.25);
    near(registered[0]->gradients()[0], -1.5);
    near(registered[1]->gradients()[0], -3.0);
}

void test_validation_is_read_only()
{
    ToyImageReconstructionTask task(1, 1, 1);
    const auto batch = one_pixel(0.5);
    task.layer().weights().values()[0] = 1.0;
    task.layer().bias().values()[0] = 0.0;

    auto params = task.parameters();
    params[0]->gradients()[0] = 8.0;
    params[1]->gradients()[0] = 9.0;

    near(task.validation_loss(batch), 0.0);
    near(task.validation_loss(batch), 0.0);
    near(params[0]->gradients()[0], 8.0);
    near(params[1]->gradients()[0], 9.0);
    near(params[0]->values()[0], 1.0);
    near(params[1]->values()[0], 0.0);
}

void test_multiple_samples_and_features()
{
    ToyImageReconstructionTask task(1, 1, 2);
    Tensor images({2, 1, 1, 2});
    images[0] = 0.0;
    images[1] = 0.5;
    images[2] = 1.0;
    images[3] = 0.5;

    const ImageBatch batch(std::move(images), {"image-a", "image-b"});
    const double expected_loss = (0.0 + 0.25 + 1.0 + 0.25) / 4.0;
    near(task.validation_loss(batch), expected_loss);
    near(task.train_microbatch(batch, 2.0), expected_loss);

    const auto params = task.parameters();
    // For output component 0, inputs (0,0.5), (1,0.5)
    // dMSE/dy = 2*(y-x)/4 * scale(2) = -x.
    // Weight gradients: [-1, -0.5], bias: -1.
    near(params[0]->gradients()[0], -1.0);
    near(params[0]->gradients()[1], -0.5);
    near(params[1]->gradients()[0], -1.0);
}

void test_invalid_and_atomic_failure()
{
    ToyImageReconstructionTask task(1, 1, 1);
    auto params = task.parameters();
    params[0]->gradients()[0] = 7.0;
    params[1]->gradients()[0] = 8.0;

    const auto batch = one_pixel(0.5);

    rejects<std::invalid_argument>([&] {
        (void)task.train_microbatch(batch, -1.0);
    });
    rejects<std::invalid_argument>([&] {
        (void)task.train_microbatch(
            batch, std::numeric_limits<double>::quiet_NaN());
    });

    // Finite but unrepresentable multiplier must not partly accumulate.
    rejects<std::overflow_error>([&] {
        (void)task.train_microbatch(
            batch, std::numeric_limits<double>::max());
    });

    near(params[0]->gradients()[0], 7.0);
    near(params[1]->gradients()[0], 8.0);
    task.reset_transient_state();

    ToyImageReconstructionTask wrong_shape(1, 2, 1);
    rejects<std::invalid_argument>([&] {
        (void)wrong_shape.validation_loss(batch);
    });

    rejects<std::invalid_argument>([] {
        ToyImageReconstructionTask bad(2, 1, 1);
    });
    rejects<std::invalid_argument>([] {
        ToyImageReconstructionTask bad(1, 0, 1);
    });
}

void test_deterministic_replay()
{
    ToyImageReconstructionTask first(1, 1, 1);
    ToyImageReconstructionTask second(1, 1, 1);
    const auto batch = one_pixel(0.25);

    near(first.train_microbatch(batch, 1.0),
         second.train_microbatch(batch, 1.0));
    const auto p1 = first.parameters();
    const auto p2 = second.parameters();
    for (std::size_t p = 0; p < p1.size(); ++p) {
        require(p1[p]->gradients().size() ==
                p2[p]->gradients().size(),
                "Deterministic gradient shape changed");
        for (std::size_t i = 0; i < p1[p]->gradients().size(); ++i) {
            near(p1[p]->gradients()[i], p2[p]->gradients()[i], 0.0);
        }
    }
}

} // namespace

int main()
{
    try {
        test_known_answer_and_scaled_backprop();
        test_validation_is_read_only();
        test_multiple_samples_and_features();
        test_invalid_and_atomic_failure();
        test_deterministic_replay();
        std::cout << "[PASS] NN.1.07E.02 image reconstruction task\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] NN.1.07E.02: "
                  << error.what() << '\n';
        return 1;
    }
}
