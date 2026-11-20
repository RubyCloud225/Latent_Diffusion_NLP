
#include "latent/nn/Conv2D.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace latent::nn {

namespace {

std::size_t padded_extent(
    std::size_t extent,
    std::size_t padding
)
{
    const auto max =
        std::numeric_limits<std::size_t>::max();

    if (padding > (max - extent) / 2) {
        throw std::overflow_error(
            "Conv2D padded dimension overflow"
        );
    }

    return extent + 2 * padding;
}

std::size_t output_extent(
    std::size_t extent,
    std::size_t kernel,
    std::size_t stride,
    std::size_t padding
)
{
    const std::size_t padded =
        padded_extent(extent, padding);

    if (kernel > padded) {
        throw std::invalid_argument(
            "Conv2D kernel exceeds padded input"
        );
    }

    return (padded - kernel) / stride + 1;
}

} // namespace

Conv2D::Conv2D(
    std::size_t in_channels,
    std::size_t out_channels,
    std::size_t kernel_h,
    std::size_t kernel_w,
    std::size_t stride,
    std::size_t padding
)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_h_(kernel_h),
      kernel_w_(kernel_w),
      stride_(stride),
      padding_(padding),
      weights_({
          out_channels, in_channels,
          kernel_h, kernel_w
      }),
      bias_({out_channels})
{
    if (stride == 0) {
        throw std::invalid_argument(
            "Conv2D stride must be positive"
        );
    }
}

latent::core::Tensor Conv2D::forward(
    const latent::core::Tensor& input
)
{
    if (input.rank() != 4) {
        throw std::invalid_argument(
            "Conv2D requires NCHW rank-4 input"
        );
    }

    const auto& shape = input.shape();

    const std::size_t batch = shape[0];
    const std::size_t channels = shape[1];
    const std::size_t height = shape[2];
    const std::size_t width = shape[3];

    if (channels != in_channels_) {
        throw std::invalid_argument(
            "Conv2D input channel mismatch"
        );
    }

    if (!input.all_finite() ||
        !weights_.values().all_finite() ||
        !bias_.values().all_finite()) {
        throw std::invalid_argument(
            "Conv2D requires finite inputs and parameters"
        );
    }

    const std::size_t out_h = output_extent(
        height, kernel_h_, stride_, padding_
    );

    const std::size_t out_w = output_extent(
        width, kernel_w_, stride_, padding_
    );

    latent::core::Tensor output({
        batch, out_channels_, out_h, out_w
    });

    const auto& weights = weights_.values();
    const auto& bias = bias_.values();

    for (std::size_t n = 0; n < batch; ++n) {
        for (std::size_t oc = 0;
             oc < out_channels_; ++oc) {

            for (std::size_t oh = 0;
                 oh < out_h; ++oh) {

                for (std::size_t ow = 0;
                     ow < out_w; ++ow) {

                    double sum = bias[oc];

                    for (std::size_t ic = 0;
                         ic < in_channels_; ++ic) {

                        for (std::size_t kh = 0;
                             kh < kernel_h_; ++kh) {

                            const std::size_t py =
                                oh * stride_ + kh;

                            // Coordinates within the
                            // virtual padded image.
                            if (py < padding_) {
                                continue;
                            }

                            const std::size_t iy =
                                py - padding_;

                            if (iy >= height) {
                                continue;
                            }

                            for (std::size_t kw = 0;
                                 kw < kernel_w_; ++kw) {

                                const std::size_t px =
                                    ow * stride_ + kw;

                                if (px < padding_) {
                                    continue;
                                }

                                const std::size_t ix =
                                    px - padding_;

                                if (ix >= width) {
                                    continue;
                                }

                                const std::size_t input_idx =
                                    ((n * in_channels_ + ic)
                                        * height + iy)
                                        * width + ix;

                                const std::size_t weight_idx =
                                    ((oc * in_channels_ + ic)
                                        * kernel_h_ + kh)
                                        * kernel_w_ + kw;

                                sum +=
                                    input[input_idx] *
                                    weights[weight_idx];
                            }
                        }
                    }

                    if (!std::isfinite(sum)) {
                        throw std::overflow_error(
                            "Conv2D produced non-finite output"
                        );
                    }

                    const std::size_t output_idx =
                        ((n * out_channels_ + oc)
                            * out_h + oh)
                            * out_w + ow;

                    output[output_idx] = sum;
                }
            }
        }
    }

    // Cache only after the complete forward computation
    // has succeeded. Invalid input cannot corrupt the
    // previous valid training state.
    cached_input_ = input;
    cached_output_shape_ = output.shape();
    has_forward_ = true;
    return output;
}


latent::core::Tensor Conv2D::backward(
    const latent::core::Tensor& grad_output
)
{
    if (!has_forward_) {
        throw std::logic_error(
            "Conv2D backward requires a successful forward"
        );
    }

    if (grad_output.shape() != cached_output_shape_) {
        throw std::invalid_argument(
            "Conv2D output gradient shape mismatch"
        );
    }

    if (!grad_output.all_finite() ||
        !cached_input_.all_finite() ||
        !weights_.values().all_finite()) {
        throw std::invalid_argument(
            "Conv2D backward requires finite values"
        );
    }

    const auto& input_shape = cached_input_.shape();
    const auto& output_shape = cached_output_shape_;

    const std::size_t batch = input_shape[0];
    const std::size_t height = input_shape[2];
    const std::size_t width = input_shape[3];

    const std::size_t out_h = output_shape[2];
    const std::size_t out_w = output_shape[3];

    latent::core::Tensor grad_input(input_shape);
    latent::core::Tensor grad_weights(
        weights_.values().shape()
    );
    latent::core::Tensor grad_bias(
        bias_.values().shape()
    );

    const auto& weights = weights_.values();

    for (std::size_t n = 0; n < batch; ++n) {
        for (std::size_t oc = 0;
             oc < out_channels_; ++oc) {

            for (std::size_t oh = 0;
                 oh < out_h; ++oh) {

                for (std::size_t ow = 0;
                     ow < out_w; ++ow) {

                    const std::size_t output_idx =
                        ((n * out_channels_ + oc)
                            * out_h + oh)
                            * out_w + ow;

                    const double g = grad_output[output_idx];

                    // Bias gradient: accumulate upstream
                    // gradients across batch and space.
                    grad_bias[oc] += g;

                    for (std::size_t ic = 0;
                         ic < in_channels_; ++ic) {

                        for (std::size_t kh = 0;
                             kh < kernel_h_; ++kh) {

                            const std::size_t py =
                                oh * stride_ + kh;

                            if (py < padding_) {
                                continue;
                            }

                            const std::size_t iy =
                                py - padding_;

                            if (iy >= height) {
                                continue;
                            }

                            for (std::size_t kw = 0;
                                 kw < kernel_w_; ++kw) {

                                const std::size_t px =
                                    ow * stride_ + kw;

                                if (px < padding_) {
                                    continue;
                                }

                                const std::size_t ix =
                                    px - padding_;

                                if (ix >= width) {
                                    continue;
                                }

                                const std::size_t input_idx =
                                    ((n * in_channels_ + ic)
                                        * height + iy)
                                        * width + ix;

                                const std::size_t weight_idx =
                                    ((oc * in_channels_ + ic)
                                        * kernel_h_ + kh)
                                        * kernel_w_ + kw;

                                // dL/dW = upstream * input
                                grad_weights[weight_idx] +=
                                    g * cached_input_[input_idx];

                                // dL/dX = upstream * weight
                                grad_input[input_idx] +=
                                    g * weights[weight_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    if (!grad_input.all_finite() ||
        !grad_weights.all_finite() ||
        !grad_bias.all_finite()) {
        throw std::overflow_error(
            "Conv2D backward produced non-finite gradients"
        );
    }

    // Stage both updates before committing either one.
    // This prevents a failure in bias accumulation from
    // leaving the weight gradients partially updated.
    latent::core::Tensor updated_weights =
        weights_.gradients();

    latent::core::Tensor updated_bias =
        bias_.gradients();

    if (!updated_weights.all_finite() ||
        !updated_bias.all_finite()) {
        throw std::invalid_argument(
            "Conv2D stored gradients must be finite"
        );
    }

    for (std::size_t i = 0;
         i < updated_weights.size(); ++i) {
        updated_weights[i] += grad_weights[i];
    }

    for (std::size_t i = 0;
         i < updated_bias.size(); ++i) {
        updated_bias[i] += grad_bias[i];
    }

    if (!updated_weights.all_finite() ||
        !updated_bias.all_finite()) {
        throw std::overflow_error(
            "Conv2D gradient accumulation overflow"
        );
    }

    weights_.gradients() = std::move(updated_weights);
    bias_.gradients() = std::move(updated_bias);

    return grad_input;
}


std::vector<Parameter*> Conv2D::parameters()
{
    return {&weights_, &bias_};
}

Parameter& Conv2D::weights() noexcept
{
    return weights_;
}

Parameter& Conv2D::bias() noexcept
{
    return bias_;
}

} // namespace latent::nn
