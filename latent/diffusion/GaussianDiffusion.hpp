#pragma once

#include "latent/core/Tensor.hpp"
#include "latent/diffusion/NoiseSchedule.hpp"

#include <cstddef>

namespace latent::diffusion {

/**
 * P0.7: closed-form forward DDPM diffusion.
 *
 * q_sample(z0,t,epsilon) = sqrt(alpha_bar[t])*z0
 *                        + sqrt(1-alpha_bar[t])*epsilon.
 * The caller owns and supplies epsilon; no implicit RNG or mutation.
 *
 * The referenced schedule must outlive this instance.
 */
class GaussianDiffusion final {
public:
    explicit GaussianDiffusion(const NoiseSchedule& schedule) noexcept;

    latent::core::Tensor q_sample(
        const latent::core::Tensor& z0,
        std::size_t timestep,
        const latent::core::Tensor& noise
    ) const;

private:
    const NoiseSchedule& schedule_;
};

} // namespace latent::diffusion
