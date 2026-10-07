#pragma once

#include "../../experimental/CliffordMultivector.hpp"
#include "../../experimental/CliffordProjection.hpp"

#include <vector>

namespace latent::backend::metal {

class CliffordProjectionMetal {
public:
    using Projection =
        latent::experimental::CliffordProjection;

    using Matrix =
        Projection::Matrix;
    using BackwardResult =
        Projection::CliffordBackwardResult;

    /**
     * Execute the same 64 -> 8 Clifford projection
     * as CliffordProjection::forward(), but using Metal.
     *
     * This validation implementation copies the result
     * back to CPU memory.
     */
    static latent::experimental::CliffordMultivector forward(
        const std::vector<float>& input,
        const Matrix& weights
    );

    static BackwardResult backward(
        const std::vector<float>& input,
        const std::array<float, Projection::BladeCount>& grad_output,
        const Matrix& weights
    );
};

} // namespace latent::backend::metal