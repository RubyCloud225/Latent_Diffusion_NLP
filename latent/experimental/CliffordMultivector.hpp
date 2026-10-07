#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace latent::experimental {

struct CliffordMultivector {
    static constexpr std::size_t BladeCount = 8;

    /**
     * Cl(3,0) coefficient ordering:
     *
     * [0] scalar
     * [1] e1
     * [2] e2
     * [3] e3
     * [4] e12
     * [5] e13
     * [6] e23
     * [7] e123
     */
    std::array<float, BladeCount> fp32{};

    /**
     * Raw IEEE-754 FP16 representations
     * of the same eight coefficients.
     */
    std::array<std::uint16_t, BladeCount> fp16{};
};

} // namespace latent::experimental