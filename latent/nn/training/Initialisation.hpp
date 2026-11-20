
#pragma once

#include "latent/nn/Conv2D.hpp"
#include "latent/nn/Linear.hpp"

#include <cstdint>
#include <string>

namespace latent::nn::training {

/**
 * @brief Supported FP64 weight initialisation schemes.
 */
enum class Initialisation {
    HeNormal,
    XavierNormal
};

/**
 * @brief Reproducibility metadata for one layer.
 */
struct InitialisationRecord {
    std::uint64_t master_seed = 0;
    std::uint64_t derived_seed = 0;

    std::string layer_id;
    std::string algorithm_id = "initialisation_v1";

    Initialisation method = Initialisation::HeNormal;

    std::size_t fan_in = 0;
    std::size_t fan_out = 0;

    double standard_deviation = 0.0;
};

/**
 * @brief Initialise production FP64 layer parameters.
 *
 * @details Derives a reproducible per-layer random stream
 * from a master seed and stable layer identifier.
 *
 * Samples Gaussian kernel/weight values using the existing
 * RandomGenerator and resets biases to zero.
 *
 * @note Parameter gradients are reset on successful
 *       initialisation.
 * @warning Call before creating or stepping the optimiser,
 *          not during an active training run.
 */
class ParameterInitialiser final {
public:
    /**
     * @brief Construct a seeded initialiser.
     * @param master_seed Reproducible training seed.
     */
    explicit ParameterInitialiser(
        std::uint64_t master_seed
    ) noexcept;

    /**
     * @brief Initialise a fully connected layer.
     * @param layer Target layer.
     * @param layer_id Stable, non-empty layer identifier.
     * @param method Chosen distribution.
     * @return Initialisation metadata.
     */
    InitialisationRecord initialise(
        Linear& layer,
        const std::string& layer_id,
        Initialisation method
    ) const;

    /**
     * @brief Initialise a convolution layer.
     * @param layer Target layer.
     * @param layer_id Stable, non-empty layer identifier.
     * @param method Chosen distribution.
     * @return Initialisation metadata.
     */
    InitialisationRecord initialise(
        Conv2D& layer,
        const std::string& layer_id,
        Initialisation method
    ) const;

private:
    std::uint64_t master_seed_;
};

} // namespace latent::nn::training
