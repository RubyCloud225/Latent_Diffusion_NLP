#include "../../latent/experimental/CliffordProjection.hpp"

#include "../../latent/core/FP16.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <utility>

namespace latent::experimental {

CliffordProjection::CliffordProjection(
    Matrix weights
)
    : weights_(std::move(weights))
{
}

CliffordProjection::CliffordProjection(
    std::uint32_t seed
)
    : weights_(make_xavier_weights(seed))
{
}

CliffordProjection::Matrix
CliffordProjection::make_xavier_weights(
    std::uint32_t seed
)
{
    Matrix weights{};

    constexpr float fan_in =
        static_cast<float>(
            SourceDimensions
        );

    constexpr float fan_out =
        static_cast<float>(
            BladeCount
        );

    const float limit =
        std::sqrt(
            6.0f /
            (fan_in + fan_out)
        );

    std::mt19937 generator(seed);

    std::uniform_real_distribution<float>
        distribution(
            -limit,
            limit
        );

    for (auto& weight : weights) {
        weight =
            distribution(generator);
    }

    return weights;
}

CliffordMultivector
CliffordProjection::forward(
    const std::vector<float>& input
) const
{
    if (
        input.size() != SourceDimensions
    ) {
        throw std::invalid_argument(
            "CliffordProjection::forward expected "
            "a 64-dimensional input."
        );
    }

    CliffordMultivector result{};

    for (
        std::size_t blade = 0;
        blade < BladeCount;
        ++blade
    ) {
        float coefficient = 0.0f;

        for (
            std::size_t dimension = 0;
            dimension < SourceDimensions;
            ++dimension
        ) {
            const std::size_t index =
                blade * SourceDimensions +
                dimension;

            coefficient +=
                weights_[index] *
                input[dimension];
        }

        result.fp32[blade] =
            coefficient;

        result.fp16[blade] =
            latent::core::float_to_fp16(
                coefficient
            );
    }

    return result;
}

CliffordProjection::CliffordBackwardResult
CliffordProjection::backward(
    const std::vector<float>& input,
    const std::array<float, BladeCount>& grad_output
) const
{
    if (
        input.size() != SourceDimensions
    ) {
        throw std::invalid_argument(
            "CliffordProjection::backward expected "
            "a 64-dimensional input."
        );
    }

    CliffordBackwardResult result;
    result.grad_input.resize(
        SourceDimensions,
        0.0f
    );

    for (
        std::size_t blade = 0;
        blade < BladeCount;
        ++blade
    ) {
        for (
            std::size_t dimension = 0;
            dimension < SourceDimensions;
            ++dimension
        ) {
            const std::size_t index =
                blade * SourceDimensions +
                dimension;

            result.grad_weights[index] =
                grad_output[blade] *
                input[dimension];

            result.grad_input[dimension] +=
                weights_[index] *
                grad_output[blade];
        }
    }

    return result;
}

std::vector<float>
CliffordProjection::reconstruct(
    const CliffordMultivector& multivector
) const
{
    std::vector<float> output(
        SourceDimensions,
        0.0f
    );

    for (
        std::size_t dimension = 0;
        dimension < SourceDimensions;
        ++dimension
    ) {
        float value = 0.0f;

        for (
            std::size_t blade = 0;
            blade < BladeCount;
            ++blade
        ) {
            const std::size_t index =
                blade * SourceDimensions +
                dimension;

            value +=
                weights_[index] *
                multivector.fp32[blade];
        }

        output[dimension] =
            value;
    }

    return output;
}

CliffordProjection::Matrix&
CliffordProjection::weights() noexcept
{
    return weights_;
}

const CliffordProjection::Matrix&
CliffordProjection::weights() const noexcept
{
    return weights_;
}
std::vector<float>
CliffordProjection::weights_vector() const
{
    return std::vector<float>(
        weights_.begin(),
        weights_.end()
    );
}


void CliffordProjection::set_weights(
    const std::vector<float>& weights
)
{
    if (
        weights.size() !=
        weights_.size()
    ) {
        throw std::invalid_argument(
            "CliffordProjection::set_weights "
            "received incorrect parameter count."
        );
    }

    std::copy(
        weights.begin(),
        weights.end(),
        weights_.begin()
    );
}

} // namespace latent::experimental