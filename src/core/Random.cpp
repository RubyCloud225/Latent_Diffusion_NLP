
#include "latent/core/Random.hpp"

#include <cmath>
#include <cstdint>

namespace latent::core {

namespace {

// Number of bits in a double's significand.
constexpr double kTwoPow53 = 9007199254740992.0;

// High-precision constant avoids reliance on M_PI.
constexpr double kTwoPi =
    6.2831853071795864769252867665590057683943388;

} // namespace

RandomGenerator::RandomGenerator(std::uint64_t seed)
    : engine_(seed)
{
}

void RandomGenerator::reseed(std::uint64_t seed)
{
    engine_.seed(seed);

    // Cached Gaussian values belong to the old
    // sequence and must not survive reseeding.
    has_spare_ = false;
    spare_ = 0.0;
}

double RandomGenerator::uniform()
{
    // Use the upper 53 random bits to construct
    // a value representable in [0, 1).
    //
    // No standard-library distribution is used,
    // making the integer-to-double mapping explicit.
    const std::uint64_t bits = engine_() >> 11;

    return static_cast<double>(bits) / kTwoPow53;
}

double RandomGenerator::gaussian()
{
    if (has_spare_) {
        has_spare_ = false;
        return spare_;
    }

    // Box-Muller requires u1 in (0, 1].
    // The midpoint mapping avoids log(0).
    const std::uint64_t bits = engine_() >> 11;

    const double u1 =
        (static_cast<double>(bits) + 0.5)
        / kTwoPow53;

    // Independent uniform angle in [0, 2*pi).
    const double u2 = uniform();

    const double radius =
        std::sqrt(-2.0 * std::log(u1));

    const double angle = kTwoPi * u2;

    const double z0 = radius * std::cos(angle);
    const double z1 = radius * std::sin(angle);

    spare_ = z1;
    has_spare_ = true;

    return z0;
}

Tensor RandomGenerator::gaussian_tensor(
    const Tensor::Shape& shape
)
{
    // Tensor validates rank, dimensions and size.
    Tensor result(shape);

    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = gaussian();
    }

    return result;
}

} // namespace latent::core
