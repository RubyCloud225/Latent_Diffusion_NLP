#pragma once

#include "latent/core/Tensor.hpp"
#include <cstdint>
#include <random>

namespace latent::core {

/**
 * @brief Seeded CPU uniform and standard-Gaussian random generator.
 *
 * @details Each instance owns an independent MT19937-64 engine and
 * Box-Muller spare-value cache. Identical seeds and call sequences are
 * reproducible on the same implementation. This is not cryptographic RNG.
 */
class RandomGenerator final {
public:
    /** @brief Initialise the random stream with an explicit seed.
     * @param seed 64-bit initial seed.
     */
    explicit RandomGenerator(std::uint64_t seed);

    /** @brief Sample a uniform double.
     * @return Value in the half-open interval [0,1).
     */
    double uniform();

    /** @brief Sample from the standard normal distribution.
     * @return Gaussian value of nominal mean zero and variance one.
     */
    double gaussian();

    /** @brief Generate an owning Tensor of Gaussian samples.
     * @param shape Tensor dimensions, validated by Tensor.
     * @return Tensor with independent sequential Gaussian samples.
     */
    Tensor gaussian_tensor(const Tensor::Shape& shape);

    /** @brief Reset both engine and cached Gaussian sample.
     * @param seed New random seed.
     */
    void reseed(std::uint64_t seed);

private:
    std::mt19937_64 engine_;
    bool has_spare_ = false;
    double spare_ = 0.0;
};

} // namespace latent::core
