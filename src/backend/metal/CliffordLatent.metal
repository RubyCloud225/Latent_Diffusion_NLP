#include <metal_stdlib>

using namespace metal;

constant uint SOURCE_DIMENSIONS = 64;
constant uint BLADE_COUNT = 8;

/**
 * Compute:
 *
 *     c = P x
 *
 * where:
 *
 *     x : 64-dimensional latent
 *     P : 8 x 64 trainable projection matrix
 *     c : 8 Clifford coefficients
 *
 * One GPU thread computes one Clifford coefficient.
 */
kernel void clifford_projection_forward(
    device const float* input
        [[buffer(0)]],

    device const float* weights
        [[buffer(1)]],

    device float* output_fp32
        [[buffer(2)]],

    device half* output_fp16
        [[buffer(3)]],

    uint blade
        [[thread_position_in_grid]]
)
{
    if (blade >= BLADE_COUNT) {
        return;
    }

    float coefficient = 0.0f;

    const uint row_offset =
        blade * SOURCE_DIMENSIONS;

    for (
        uint dimension = 0;
        dimension < SOURCE_DIMENSIONS;
        ++dimension
    ) {
        coefficient +=
            weights[
                row_offset + dimension
            ] *
            input[dimension];
    }

    output_fp32[blade] =
        coefficient;

    output_fp16[blade] =
        half(coefficient);
}
/**
 * Compute:
 *
 * dL/dP = g x^T
 *
 * There are 8 * 64 = 512 weight gradients.
 * One GPU thread computes one gradient.
 */
kernel void clifford_projection_backward_weights(
    device const float* input
        [[buffer(0)]],

    device const float* grad_output
        [[buffer(1)]],

    device float* grad_weights
        [[buffer(2)]],

    uint index
        [[thread_position_in_grid]]
)
{
    constexpr uint WEIGHT_COUNT =
        BLADE_COUNT * SOURCE_DIMENSIONS;

    if (index >= WEIGHT_COUNT) {
        return;
    }

    const uint blade =
        index / SOURCE_DIMENSIONS;

    const uint dimension =
        index % SOURCE_DIMENSIONS;

    grad_weights[index] =
        grad_output[blade] *
        input[dimension];
}


/**
 * Compute:
 *
 * dL/dx = P^T g
 *
 * One GPU thread computes one input dimension.
 */
kernel void clifford_projection_backward_input(
    device const float* weights
        [[buffer(0)]],

    device const float* grad_output
        [[buffer(1)]],

    device float* grad_input
        [[buffer(2)]],

    uint dimension
        [[thread_position_in_grid]]
)
{
    if (dimension >= SOURCE_DIMENSIONS) {
        return;
    }

    float gradient = 0.0f;

    for (
        uint blade = 0;
        blade < BLADE_COUNT;
        ++blade
    ) {
        const uint index =
            blade * SOURCE_DIMENSIONS +
            dimension;

        gradient +=
            weights[index] *
            grad_output[blade];
    }

    grad_input[dimension] =
        gradient;
}