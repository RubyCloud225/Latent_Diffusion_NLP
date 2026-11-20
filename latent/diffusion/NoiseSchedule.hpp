#pragma once

#include <cstddef>
#include <vector>

namespace latent::diffusion {

/**
 * @brief Input configuration for a linearly interpolated DDPM schedule.
 *
 * @details Requires timesteps >= 1 and 0 < beta_start <= beta_end < 1.
 * A one-timestep schedule uses beta_start.
 */
struct ScheduleConfig {
    std::size_t timesteps; ///< Number of discrete noise levels.
    double beta_start;     ///< Initial noise-variance coefficient.
    double beta_end;       ///< Final noise-variance coefficient.
};

/**
 * @brief Immutable precomputed FP64 DDPM forward-process schedule.
 *
 * @details Stores beta[t], alpha[t]=1-beta[t], cumulative alpha_bar[t],
 * sqrt(alpha_bar[t]) and sqrt(1-alpha_bar[t]) for zero-based t.
 */
class NoiseSchedule final {
public:
    /** @brief Validate configuration and precompute all coefficients.
     * @throws std::invalid_argument If configuration is invalid.
     * @throws std::overflow_error If cumulative alpha loses validity.
     */
    explicit NoiseSchedule(const ScheduleConfig& config);

    /** @return Total configured timesteps. */
    std::size_t timesteps() const noexcept;

    /** @brief Return beta_t. @throws std::out_of_range Invalid timestep. */
    double beta(std::size_t t) const;
    /** @brief Return alpha_t. @throws std::out_of_range Invalid timestep. */
    double alpha(std::size_t t) const;
    /** @brief Return cumulative alpha_bar_t.
     * @throws std::out_of_range Invalid timestep.
     */
    double alpha_bar(std::size_t t) const;
    /** @brief Return sqrt(alpha_bar_t).
     * @throws std::out_of_range Invalid timestep.
     */
    double sqrt_alpha_bar(std::size_t t) const;
    /** @brief Return sqrt(1 - alpha_bar_t).
     * @throws std::out_of_range Invalid timestep.
     */
    double sqrt_one_minus_alpha_bar(std::size_t t) const;

private:
    std::vector<double> beta_;
    std::vector<double> alpha_;
    std::vector<double> alpha_bar_;
    std::vector<double> sqrt_alpha_bar_;
    std::vector<double> sqrt_one_minus_alpha_bar_;
};

} // namespace latent::diffusion
