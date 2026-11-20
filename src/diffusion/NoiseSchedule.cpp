
#include "latent/diffusion/NoiseSchedule.hpp"

#include <cmath>
#include <stdexcept>

namespace latent::diffusion {

namespace {

void validate_config(const ScheduleConfig& config)
{
    if (config.timesteps == 0) {
        throw std::invalid_argument(
            "NoiseSchedule requires at least one timestep"
        );
    }

    if (!std::isfinite(config.beta_start) ||
        !std::isfinite(config.beta_end)) {
        throw std::invalid_argument(
            "Beta endpoints must be finite"
        );
    }

    if (!(config.beta_start > 0.0) ||
        !(config.beta_end < 1.0) ||
        config.beta_start > config.beta_end) {
        throw std::invalid_argument(
            "Require 0 < beta_start <= beta_end < 1"
        );
    }
}

void validate_coefficient(double value)
{
    if (!std::isfinite(value)) {
        throw std::overflow_error(
            "Non-finite diffusion coefficient"
        );
    }
}

} // namespace

NoiseSchedule::NoiseSchedule(const ScheduleConfig& config)
{
    validate_config(config);

    beta_.reserve(config.timesteps);
    alpha_.reserve(config.timesteps);
    alpha_bar_.reserve(config.timesteps);
    sqrt_alpha_bar_.reserve(config.timesteps);
    sqrt_one_minus_alpha_bar_.reserve(config.timesteps);

    double cumulative_alpha = 1.0;

    for (std::size_t t = 0; t < config.timesteps; ++t) {
        const double fraction =
            config.timesteps == 1
                ? 0.0
                : static_cast<double>(t) /
                  static_cast<double>(config.timesteps - 1);

        const double beta_t =
            (1.0 - fraction) * config.beta_start
            + fraction * config.beta_end;

        const double alpha_t = 1.0 - beta_t;

        cumulative_alpha *= alpha_t;

        // Reject schedules where cumulative precision
        // has collapsed to zero (numerical underflow).
        if (!(cumulative_alpha > 0.0) ||
            !std::isfinite(cumulative_alpha)) {
            throw std::overflow_error(
                "Cumulative alpha underflow"
            );
        }

        const double root_alpha =
            std::sqrt(cumulative_alpha);

        // Protect against tiny roundoff excursions.
        const double one_minus_alpha =
            std::fmax(0.0, 1.0 - cumulative_alpha);

        const double root_one_minus_alpha =
            std::sqrt(one_minus_alpha);

        validate_coefficient(beta_t);
        validate_coefficient(alpha_t);
        validate_coefficient(root_alpha);
        validate_coefficient(root_one_minus_alpha);

        beta_.push_back(beta_t);
        alpha_.push_back(alpha_t);
        alpha_bar_.push_back(cumulative_alpha);
        sqrt_alpha_bar_.push_back(root_alpha);
        sqrt_one_minus_alpha_bar_.push_back(
            root_one_minus_alpha
        );
    }
}

std::size_t NoiseSchedule::timesteps() const noexcept
{
    return beta_.size();
}

double NoiseSchedule::beta(std::size_t t) const
{
    return beta_.at(t);
}

double NoiseSchedule::alpha(std::size_t t) const
{
    return alpha_.at(t);
}

double NoiseSchedule::alpha_bar(std::size_t t) const
{
    return alpha_bar_.at(t);
}

double NoiseSchedule::sqrt_alpha_bar(std::size_t t) const
{
    return sqrt_alpha_bar_.at(t);
}

double NoiseSchedule::sqrt_one_minus_alpha_bar(
    std::size_t t
) const
{
    return sqrt_one_minus_alpha_bar_.at(t);
}

} // namespace latent::diffusion
