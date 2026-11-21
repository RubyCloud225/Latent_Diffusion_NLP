
#include "latent/nn/Linear.hpp"

#include <cmath>
#include <stdexcept>

namespace latent::nn {

namespace {

void require_finite(
    const latent::core::Tensor& tensor
) {
    if (!tensor.all_finite()) {
        throw std::invalid_argument(
            "Linear requires finite Tensor values"
        );
    }
}

} // namespace

Linear::Linear(
    std::size_t in_features,
    std::size_t out_features
)
    : in_features_(in_features),
      out_features_(out_features),
      weights_({out_features, in_features}),
      bias_({out_features})
{
    if (in_features == 0 || out_features == 0) {
        throw std::invalid_argument(
            "Linear dimensions must be positive"
        );
    }
}

latent::core::Tensor Linear::forward(
    const latent::core::Tensor& input
)
{
    if (input.rank() != 1 && input.rank() != 2) {
        throw std::invalid_argument(
            "Linear requires rank-1 or rank-2 input"
        );
    }

    if (input.shape().back() != in_features_) {
        throw std::invalid_argument(
            "Linear input feature mismatch"
        );
    }

    require_finite(input);
    require_finite(weights_.values());
    require_finite(bias_.values());

    const std::size_t batch =
        input.rank() == 1 ? 1 : input.shape()[0];

    latent::core::Tensor output(
        input.rank() == 1
            ? latent::core::Tensor::Shape{out_features_}
            : latent::core::Tensor::Shape{
                  batch, out_features_
              }
    );

    const auto& w = weights_.values();
    const auto& b = bias_.values();

    for (std::size_t n = 0; n < batch; ++n) {
        for (std::size_t o = 0;
             o < out_features_; ++o) {

            double value = b[o];

            for (std::size_t i = 0;
                 i < in_features_; ++i) {

                value +=
                    w[o * in_features_ + i] *
                    input[n * in_features_ + i];
            }

            if (!std::isfinite(value)) {
                throw std::overflow_error(
                    "Linear forward produced non-finite output"
                );
            }

            output[n * out_features_ + o] = value;
        }
    }

    // Commit the forward cache only after successful
    // validation and numerical computation.
    cached_input_ = input;
    has_forward_ = true;

    return output;
}

latent::core::Tensor Linear::backward(
    const latent::core::Tensor& grad_output
)
{
    if (!has_forward_) {
        throw std::logic_error(
            "Linear backward requires forward"
        );
    }

    const std::size_t batch =
        cached_input_.rank() == 1
            ? 1 : cached_input_.shape()[0];

    const latent::core::Tensor::Shape expected =
        cached_input_.rank() == 1
            ? latent::core::Tensor::Shape{out_features_}
            : latent::core::Tensor::Shape{
                  batch, out_features_
              };

    if (grad_output.shape() != expected) {
        throw std::invalid_argument(
            "Linear output gradient shape mismatch"
        );
    }

    require_finite(grad_output);
    require_finite(weights_.values());

    latent::core::Tensor grad_input(
        cached_input_.shape()
    );

    latent::core::Tensor grad_weights(
        weights_.values().shape()
    );

    latent::core::Tensor grad_bias(
        bias_.values().shape()
    );

    const auto& w = weights_.values();

    for (std::size_t n = 0; n < batch; ++n) {
        for (std::size_t o = 0;
             o < out_features_; ++o) {

            const double g =
                grad_output[n * out_features_ + o];

            grad_bias[o] += g;

            for (std::size_t i = 0;
                 i < in_features_; ++i) {

                const std::size_t wi =
                    o * in_features_ + i;

                grad_weights[wi] +=
                    g * cached_input_[n * in_features_ + i];

                grad_input[n * in_features_ + i] +=
                    g * w[wi];
            }
        }
    }

    if (!grad_input.all_finite() ||
        !grad_weights.all_finite() ||
        !grad_bias.all_finite()) {
        throw std::overflow_error(
            "Linear backward produced non-finite gradients"
        );
    }

    weights_.accumulate_gradient(grad_weights);
    bias_.accumulate_gradient(grad_bias);

    return grad_input;
}

std::vector<Parameter*> Linear::parameters()
{
    return {&weights_, &bias_};
}

Parameter& Linear::weights() noexcept
{
    return weights_;
}

const Parameter& Linear::weights() const noexcept
{
    return weights_;
}

Parameter& Linear::bias() noexcept
{
    return bias_;
}

const Parameter& Linear::bias() const noexcept
{
    return bias_;
}

std::size_t Linear::in_features() const noexcept
{
    return in_features_;
}

std::size_t Linear::out_features() const noexcept
{
    return out_features_;
}

} // namespace latent::nn
