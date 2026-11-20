
#pragma once

#include <cstddef>
#include <vector>

namespace latent::diffusion {

/**
 * Configuration for a linear DDPM noise schedule.
 *
 * Timesteps are zero-based: [0, timesteps).
 * Beta endpoints must satisfy 0 < beta_start
 * <= beta_end < 1.
 */
struct ScheduleConfig {
    std::size_t timesteps;
    double beta_start;
    double beta_end;
};

/**
 * P0.6 — Precomputed FP64 diffusion coefficients.
 *
 * Immutable after construction. Does not generate noise
 * or perform forward/reverse diffusion.
 */
class NoiseSchedule final {
public:
    explicit NoiseSchedule(const ScheduleConfig& config);

    std::size_t timesteps() const noexcept;

    double beta(std::size_t t) const;
    double alpha(std::size_t t) const;
    double alpha_bar(std::size_t t) const;

    double sqrt_alpha_bar(std::size_t t) const;
    double sqrt_one_minus_alpha_bar(std::size_t t) const;

private:
    std::vector<double> beta_;
    std::vector<double> alpha_;
    std::vector<double> alpha_bar_;
    std::vector<double> sqrt_alpha_bar_;
    std::vector<double> sqrt_one_minus_alpha_bar_;
};

} // namespace latent::diffusion
