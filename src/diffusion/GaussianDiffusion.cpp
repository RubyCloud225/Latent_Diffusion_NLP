#include "latent/diffusion/GaussianDiffusion.hpp"

#include <cmath>
#include <stdexcept>

namespace latent::diffusion {

GaussianDiffusion::GaussianDiffusion(const NoiseSchedule& schedule) noexcept
    : schedule_(schedule)
{
}

latent::core::Tensor GaussianDiffusion::q_sample(
    const latent::core::Tensor& z0,
    std::size_t timestep,
    const latent::core::Tensor& noise
) const
{
    if (timestep >= schedule_.timesteps()) {
        throw std::out_of_range("Forward diffusion timestep out of range");
    }
    if (!z0.same_shape(noise)) {
        throw std::invalid_argument("Input and noise Tensor shapes must match");
    }
    if (!z0.all_finite() || !noise.all_finite()) {
        throw std::invalid_argument("Forward diffusion requires finite inputs");
    }

    const double signal_scale = schedule_.sqrt_alpha_bar(timestep);
    const double noise_scale = schedule_.sqrt_one_minus_alpha_bar(timestep);

    latent::core::Tensor result(z0.shape());
    for (std::size_t i = 0; i < z0.size(); ++i) {
        const double value = signal_scale * z0[i] + noise_scale * noise[i];
        if (!std::isfinite(value)) {
            throw std::overflow_error("Forward diffusion produced a non-finite result");
        }
        result[i] = value;
    }
    return result;
}

} // namespace latent::diffusion
