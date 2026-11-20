#pragma once

#include "latent/core/Tensor.hpp"
#include "latent/diffusion/NoiseSchedule.hpp"

#include <cstddef>

namespace latent::diffusion {

/**
 * @brief Coefficients of q(z_{t-1} | z_t, predicted z_0).
 *
 * @note At t=0, posterior_variance=0, mean_coefficient_x0=1
 *       and mean_coefficient_xt=0 (terminal reverse step).
 */
struct ReverseCoefficients {
    double posterior_variance;    ///< Conditional variance for a reverse step.
    double mean_coefficient_x0;   ///< Multiplier for estimated clean latent.
    double mean_coefficient_xt;   ///< Multiplier for current noisy latent.
};

/**
 * @brief FP64 DDPM forward and deterministic reverse mathematics.
 *
 * Forward: z_t = sqrt(alpha_bar_t)*z_0 + sqrt(1-alpha_bar_t)*epsilon.
 * Reverse: estimate z_0 from externally predicted epsilon, then compute
 * the posterior mean. Sampling stochastic noise belongs to P0.9.
 *
 * @warning The referenced NoiseSchedule must outlive this object.
 */
class GaussianDiffusion final {
public:
    /** @brief Construct from an existing, immutable noise schedule.
     * @param schedule Schedule retained by reference, not copied.
     */
    explicit GaussianDiffusion(const NoiseSchedule& schedule) noexcept;

    /**
     * @brief Compute a noisy Tensor from clean input and explicit noise.
     * @param z0 Clean input Tensor.
     * @param timestep Zero-based timestep within the schedule.
     * @param noise Caller-provided epsilon Tensor, same shape as z0.
     * @return New Tensor; z0 and noise remain unchanged.
     * @throws std::out_of_range If timestep is outside the schedule.
     * @throws std::invalid_argument If shapes mismatch or inputs are not finite.
     * @throws std::overflow_error If the computed output is not finite.
     */
    latent::core::Tensor q_sample(
        const latent::core::Tensor& z0,
        std::size_t timestep,
        const latent::core::Tensor& noise
    ) const;

    /**
     * @brief Calculate posterior variance and posterior-mean multipliers.
     * @param timestep Zero-based timestep.
     * @return Coefficients for the reverse transition; variance is zero at t=0.
     * @throws std::out_of_range If timestep is outside the schedule.
     * @throws std::overflow_error If coefficients are not finite/valid.
     */
    ReverseCoefficients reverse_coefficients(std::size_t timestep) const;

    /** @brief Return the posterior variance at timestep t.
     * @param timestep Zero-based timestep.
     * @return Nonnegative posterior variance, zero at t=0.
     */
    double posterior_variance(std::size_t timestep) const;

    /**
     * @brief Reconstruct estimated clean latent from z_t and predicted noise.
     * @param zt Current noisy Tensor.
     * @param timestep Zero-based timestep.
     * @param predicted_epsilon Denoiser-predicted noise, same shape as zt.
     * @return New clean-latent estimate; inputs are not modified.
     * @throws std::invalid_argument If shapes differ or inputs are not finite.
     * @throws std::out_of_range If timestep is invalid.
     * @throws std::overflow_error If reconstruction is not finite.
     */
    latent::core::Tensor predict_x0(
        const latent::core::Tensor& zt,
        std::size_t timestep,
        const latent::core::Tensor& predicted_epsilon
    ) const;

    /**
     * @brief Calculate the deterministic reverse-process mean.
     * @param zt Current noisy Tensor.
     * @param timestep Zero-based timestep, including t=0 terminal step.
     * @param predicted_epsilon Caller-provided denoiser prediction.
     * @return Posterior mean Tensor, same shape as zt.
     * @throws std::invalid_argument If shapes differ or inputs are not finite.
     * @throws std::out_of_range If timestep is invalid.
     * @throws std::overflow_error If the result is non-finite.
     */
    latent::core::Tensor reverse_mean(
        const latent::core::Tensor& zt,
        std::size_t timestep,
        const latent::core::Tensor& predicted_epsilon
    ) const;

private:
    const NoiseSchedule& schedule_;

    void validate_reverse_inputs(
        const latent::core::Tensor& zt,
        std::size_t timestep,
        const latent::core::Tensor& predicted_epsilon
    ) const;
};

} // namespace latent::diffusion
