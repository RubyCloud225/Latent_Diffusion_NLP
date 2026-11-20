
#include "latent/nn/training/Initialisation.hpp"
#include "latent/core/Random.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace latent::nn::training {
namespace {

// SplitMix64 finalisation provides a fixed
// 64-bit integer-mixing operation.
std::uint64_t mix64(std::uint64_t x)
{
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);

    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);

    return x ^ (x >> 31);
}

std::uint64_t derive_seed(
    std::uint64_t master,
    const std::string& layer_id
)
{
    if (layer_id.empty()) {
        throw std::invalid_argument(
            "Initialisation requires a layer identifier"
        );
    }

    std::uint64_t hash =
        UINT64_C(14695981039346656037);

    // Fixed FNV-1a byte ordering.
    for (unsigned char c : layer_id) {
        hash ^= static_cast<std::uint64_t>(c);
        hash *= UINT64_C(1099511628211);
    }

    return mix64(master ^ hash);
}

double standard_deviation(
    Initialisation method,
    std::size_t fan_in,
    std::size_t fan_out
)
{
    if (fan_in == 0 || fan_out == 0) {
        throw std::invalid_argument(
            "Initialisation fan values must be positive"
        );
    }

    const double in = static_cast<double>(fan_in);
    const double out = static_cast<double>(fan_out);

    double variance = 0.0;

    switch (method) {
        case Initialisation::HeNormal:
            variance = 2.0 / in;
            break;

        case Initialisation::XavierNormal:
            // Avoid integer overflow in fan_in + fan_out.
            variance = 2.0 / in;
            variance /= (1.0 + out / in);
            break;

        default:
            throw std::invalid_argument(
                "Unsupported initialisation scheme"
            );
    }

    const double sigma = std::sqrt(variance);

    if (!std::isfinite(sigma) || sigma <= 0.0) {
        throw std::overflow_error(
            "Invalid initialisation scale"
        );
    }

    return sigma;
}

InitialisationRecord initialise_parameters(
    Parameter& weights,
    Parameter& bias,
    std::size_t fan_in,
    std::size_t fan_out,
    std::uint64_t master_seed,
    const std::string& layer_id,
    Initialisation method
)
{
    const double sigma =
        standard_deviation(method, fan_in, fan_out);

    const std::uint64_t derived =
        derive_seed(master_seed, layer_id);

    latent::core::RandomGenerator rng(derived);

    // Work in temporary tensors to avoid
    // partially updating the layer.
    latent::core::Tensor new_weights =
        weights.values();

    latent::core::Tensor new_bias =
        bias.values();

    for (std::size_t i = 0;
         i < new_weights.size(); ++i) {

        new_weights[i] = rng.gaussian() * sigma;
    }

    new_bias.fill(0.0);

    if (!new_weights.all_finite()) {
        throw std::overflow_error(
            "Non-finite initialised weights"
        );
    }

    // Commit values only after validating the
    // entire initialisation.
    weights.values() = std::move(new_weights);
    bias.values() = std::move(new_bias);

    weights.zero_grad();
    bias.zero_grad();

    InitialisationRecord record;
    record.master_seed = master_seed;
    record.derived_seed = derived;
    record.layer_id = layer_id;
    record.method = method;
    record.fan_in = fan_in;
    record.fan_out = fan_out;
    record.standard_deviation = sigma;

    return record;
}

} // namespace

ParameterInitialiser::ParameterInitialiser(
    std::uint64_t master_seed
) noexcept
    : master_seed_(master_seed)
{
}

InitialisationRecord ParameterInitialiser::initialise(
    Linear& layer,
    const std::string& layer_id,
    Initialisation method
) const
{
    return initialise_parameters(
        layer.weights(),
        layer.bias(),
        layer.in_features(),
        layer.out_features(),
        master_seed_,
        layer_id,
        method
    );
}

InitialisationRecord ParameterInitialiser::initialise(
    Conv2D& layer,
    const std::string& layer_id,
    Initialisation method
) const
{
    const auto& shape =
        layer.weights().values().shape();

    if (shape.size() != 4) {
        throw std::invalid_argument(
            "Invalid Conv2D kernel shape"
        );
    }

    const std::size_t kernel_area =
        shape[2] * shape[3];

    if (shape[1] >
            std::numeric_limits<std::size_t>::max() /
                kernel_area ||
        shape[0] >
            std::numeric_limits<std::size_t>::max() /
                kernel_area) {
        throw std::overflow_error(
            "Convolution fan calculation overflow"
        );
    }

    return initialise_parameters(
        layer.weights(),
        layer.bias(),
        shape[1] * kernel_area,
        shape[0] * kernel_area,
        master_seed_,
        layer_id,
        method
    );
}

} // namespace latent::nn::training
