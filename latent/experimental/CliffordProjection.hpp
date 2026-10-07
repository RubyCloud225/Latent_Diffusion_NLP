#pragma once

#include "../../latent/experimental/CliffordMultivector.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace latent::experimental {

class CliffordProjection {
public:
    static constexpr std::size_t SourceDimensions = 64;
    static constexpr std::size_t BladeCount = 8;

    using Matrix =
        std::array<
            float,
            BladeCount * SourceDimensions
        >;

    struct CliffordBackwardResult {
        std::vector<float> grad_input;
        Matrix grad_weights{};
    };

    explicit CliffordProjection(
        Matrix weights
    );

    explicit CliffordProjection(std::uint32_t seed = 42);

    /**
     * Project a 64D latent vector into
     * the eight coefficients of Cl(3,0).
     */
    CliffordMultivector forward(
        const std::vector<float>& input
    ) const;

    CliffordBackwardResult backward(
        const std::vector<float>& input,
        const std::array<float, BladeCount>& grad_output
    ) const;

    /**
     * Approximate reconstruction using P^T.
     *
     * Exact only for information preserved
     * by the projection.
     */
    std::vector<float> reconstruct(
        const CliffordMultivector& multivector
    ) const;

    Matrix& weights() noexcept;
    const Matrix& weights() const noexcept;
    std::vector<float> weights_vector() const;

    void set_weights(
        const std::vector<float>& weights
    );

private:
    Matrix weights_{};
    static Matrix make_xavier_weights(std::uint32_t seed);
};

} // namespace latent::experimental