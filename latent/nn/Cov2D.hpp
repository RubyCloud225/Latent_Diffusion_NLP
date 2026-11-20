
#pragma once

#include "latent/nn/Layer.hpp"

#include <cstddef>
#include <vector>

namespace latent::nn {

/**
 * @brief FP64 NCHW two-dimensional convolution layer.
 *
 * @details Implements forward cross-correlation:
 *
 * output[n,o,y,x] =
 *     bias[o] +
 *     sum(input[n,i,y',x'] * weight[o,i,ky,kx]).
 *
 * Tensor contracts:
 * - Input:   [N, Cin, H, W]
 * - Weights: [Cout, Cin, Kh, Kw]
 * - Bias:    [Cout]
 * - Output:  [N, Cout, Hout, Wout]
 *
 * Uses zero padding and a single stride for both
 * spatial dimensions.
 *
 * @note Weights and biases begin at zero.
 *       Random initialisation is a separate ticket.
 *
 * @warning Backward propagation is not implemented
 *          until NN.1.05.
 */
class Conv2D final : public Layer {
public:
    /**
     * @brief Construct a 2D convolution layer.
     *
     * @param in_channels Number of input channels.
     * @param out_channels Number of output channels.
     * @param kernel_h Kernel height.
     * @param kernel_w Kernel width.
     * @param stride Spatial stride, at least 1.
     * @param padding Symmetric zero padding.
     *
     * @throws std::invalid_argument Invalid dimensions.
     */
    Conv2D(
        std::size_t in_channels,
        std::size_t out_channels,
        std::size_t kernel_h,
        std::size_t kernel_w,
        std::size_t stride = 1,
        std::size_t padding = 0
    );

    /**
     * @brief Execute forward convolution.
     *
     * @param input Rank-4 finite NCHW Tensor.
     * @return Newly allocated FP64 output Tensor.
     *
     * @throws std::invalid_argument Invalid input shape,
     *         channel count or kernel compatibility.
     * @throws std::overflow_error Dimension or
     *         numerical overflow.
     */
    latent::core::Tensor forward(
        const latent::core::Tensor& input
    ) override;

    /**
     * @brief Backward propagation placeholder.
     *
     * @throws std::logic_error Always, until NN.1.05.
     */
    latent::core::Tensor backward(
        const latent::core::Tensor& grad_output
    ) override;

    /** @return Non-owning weight and bias parameters. */
    std::vector<Parameter*> parameters() override;

    /** @return Mutable convolution weights. */
    Parameter& weights() noexcept;

    /** @return Mutable output-channel biases. */
    Parameter& bias() noexcept;

private:
    std::size_t in_channels_;
    std::size_t out_channels_;
    std::size_t kernel_h_;
    std::size_t kernel_w_;
    std::size_t stride_;
    std::size_t padding_;

    Parameter weights_;
    Parameter bias_;
};

} // namespace latent::nn
