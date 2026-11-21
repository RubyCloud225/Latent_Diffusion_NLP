#include "latent/nn/training/ToyImageReconstructionTask.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace latent::nn::training {
namespace {

using latent::core::Tensor;

static_assert(
    std::is_nothrow_move_assignable<Tensor>::value,
    "Gradient rollback requires noexcept Tensor move assignment"
);

std::size_t image_features(
    std::size_t channels, std::size_t height, std::size_t width)
{
    if ((channels != 1 && channels != 3) ||
        height == 0 || width == 0) {
        throw std::invalid_argument("Invalid toy image shape");
    }
    if (channels > std::numeric_limits<std::size_t>::max() / height ||
        channels * height > std::numeric_limits<std::size_t>::max() / width) {
        throw std::overflow_error("Toy image feature count overflow");
    }
    return channels * height * width;
}

double mean_squared_error(const Tensor& output, const Tensor& target)
{
    if (!output.same_shape(target) || !output.all_finite()) {
        throw std::invalid_argument("Invalid reconstruction tensor");
    }

    // Independent per-element scaled sum avoids overflow in a square.
    // Since target is [0,1], a non-representable squared residual
    // is a genuine unrepresentable FP64 loss and must fail closed.
    double sum = 0.0;
    const double denominator = static_cast<double>(target.size());
    for (std::size_t i = 0; i < target.size(); ++i) {
        const double delta = output[i] - target[i];
        const double contribution = (delta / denominator) * delta;
        sum += contribution;
        if (!std::isfinite(delta) || !std::isfinite(contribution) ||
            !std::isfinite(sum) || contribution < 0.0) {
            throw std::overflow_error("Reconstruction loss overflow");
        }
    }
    return sum;
}

} // namespace

ToyImageReconstructionTask::ToyImageReconstructionTask(
    std::size_t channels, std::size_t height, std::size_t width)
    : channels_(channels),
      height_(height),
      width_(width),
      features_(image_features(channels, height, width)),
      layer_(features_, features_)
{
}

std::string ToyImageReconstructionTask::task_id() const
{
    return "toy_image_reconstruction_linear_mse_v1:" +
           std::to_string(channels_) + "x" +
           std::to_string(height_) + "x" + std::to_string(width_);
}

std::vector<latent::nn::Parameter*>
ToyImageReconstructionTask::parameters()
{
    return layer_.parameters();
}

void ToyImageReconstructionTask::require_batch(const ImageBatch& batch) const
{
    batch.validate();
    const auto& s = batch.images.shape();
    if (s[1] != channels_ || s[2] != height_ || s[3] != width_) {
        throw std::invalid_argument("Image shape incompatible with toy task");
    }
}

Tensor ToyImageReconstructionTask::flatten(const ImageBatch& batch) const
{
    Tensor result({batch.sample_count(), features_});
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = batch.images[i];
    }
    return result;
}

double ToyImageReconstructionTask::train_microbatch(
    const ImageBatch& batch, double loss_scale)
{
    require_batch(batch);
    if (!std::isfinite(loss_scale) || loss_scale < 0.0) {
        throw std::invalid_argument("Invalid reconstruction loss scale");
    }

    const auto input = flatten(batch);
    auto registered = layer_.parameters();
    std::vector<Tensor> old_gradients;
    old_gradients.reserve(registered.size());
    for (auto* parameter : registered) {
        old_gradients.push_back(parameter->gradients());
    }

    transient_valid_ = false;
    try {
        const Tensor output = layer_.forward(input);
        const double loss = mean_squared_error(output, input);
        Tensor grad_output(output.shape());

        const double divisor = static_cast<double>(output.size());
        const double factor = (2.0 * loss_scale) / divisor;
        if (!std::isfinite(factor)) {
            throw std::overflow_error("Reconstruction gradient factor overflow");
        }

        for (std::size_t i = 0; i < output.size(); ++i) {
            grad_output[i] = factor * (output[i] - input[i]);
        }
        if (!grad_output.all_finite()) {
            throw std::overflow_error("Reconstruction output gradient overflow");
        }

        transient_valid_ = true;
        (void)layer_.backward(grad_output);
        transient_valid_ = false;
        return loss;
    } catch (...) {
        // Layer::backward may accumulate weight gradients before bias
        // accumulation fails. Restore both gradients as one transaction.
        for (std::size_t i = 0; i < registered.size(); ++i) {
            registered[i]->gradients() = std::move(old_gradients[i]);
        }
        reset_transient_state();
        throw;
    }
}

double ToyImageReconstructionTask::validation_loss(
    const ImageBatch& batch) const
{
    require_batch(batch);
    // Deliberately do NOT call Linear::forward: it changes the
    // layer's cached training input and can invalidate pending backward.
    const Tensor input = flatten(batch);
    Tensor output(input.shape());
    const auto& weights = layer_.weights().values();
    const auto& bias = layer_.bias().values();
    if (!weights.all_finite() || !bias.all_finite()) {
        throw std::invalid_argument("Non-finite reconstruction parameters");
    }

    for (std::size_t n = 0; n < batch.sample_count(); ++n) {
        for (std::size_t o = 0; o < features_; ++o) {
            double value = bias[o];
            for (std::size_t i = 0; i < features_; ++i) {
                value += weights[o * features_ + i] *
                         input[n * features_ + i];
            }
            if (!std::isfinite(value)) {
                throw std::overflow_error("Validation prediction overflow");
            }
            output[n * features_ + o] = value;
        }
    }
    return mean_squared_error(output, input);
}

void ToyImageReconstructionTask::reset_transient_state() noexcept
{
    transient_valid_ = false;
}

latent::nn::Linear& ToyImageReconstructionTask::layer() noexcept
{
    return layer_;
}

} // namespace latent::nn::training
