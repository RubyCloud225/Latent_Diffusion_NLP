
#pragma once

#include "latent/core/Tensor.hpp"

#include <cstdint>
#include <random>

namespace latent::core {

/**
 * P0.5 — Deterministic CPU random number generator.
 *
 * Each instance owns its random engine and Gaussian
 * cache. Random state is never global.
 *
 * Identical seeds and identical call sequences produce
 * identical results under the same implementation.
 *
 * This generator is not cryptographically secure.
 */
class RandomGenerator final {
public:
    explicit RandomGenerator(std::uint64_t seed);

    // Return a uniformly distributed value in [0, 1).
    double uniform();

    // Return a standard-normal value: N(0, 1).
    double gaussian();

    // Produce a contiguous Gaussian Tensor.
    Tensor gaussian_tensor(const Tensor::Shape& shape);

    // Reset the complete random state.
    void reseed(std::uint64_t seed);

private:
    std::mt19937_64 engine_;

    // Cache the second Box-Muller output to avoid
    // discarding half the generated Gaussian samples.
    bool has_spare_ = false;
    double spare_ = 0.0;
};

} // namespace latent::core
