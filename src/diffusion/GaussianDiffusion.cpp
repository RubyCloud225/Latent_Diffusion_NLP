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


void GaussianDiffusion::validate_reverse_inputs(
    const latent::core::Tensor& zt,
    std::size_t timestep,
    const latent::core::Tensor& predicted_epsilon
) const
{
    if (timestep >= schedule_.timesteps()) {
        throw std::out_of_range("Reverse diffusion timestep out of range");
    }
    if (!zt.same_shape(predicted_epsilon)) {
        throw std::invalid_argument("Predicted epsilon shape must match zt");
    }
    if (!zt.all_finite() || !predicted_epsilon.all_finite()) {
        throw std::invalid_argument("Reverse diffusion requires finite inputs");
    }
}

ReverseCoefficients GaussianDiffusion::reverse_coefficients(
    std::size_t timestep
) const
{
    if (timestep >= schedule_.timesteps()) {
        throw std::out_of_range("Reverse coefficient timestep out of range");
    }

    const double beta = schedule_.beta(timestep);
    const double alpha = schedule_.alpha(timestep);
    const double alpha_bar = schedule_.alpha_bar(timestep);
    const double alpha_bar_previous =
        timestep == 0 ? 1.0 : schedule_.alpha_bar(timestep - 1);
    const double denominator = 1.0 - alpha_bar;

    if (!(denominator > 0.0) || !std::isfinite(denominator)) {
        throw std::overflow_error("Invalid reverse diffusion denominator");
    }

    ReverseCoefficients result{};
    result.posterior_variance =
        beta * (1.0 - alpha_bar_previous) / denominator;
    result.mean_coefficient_x0 =
        beta * std::sqrt(alpha_bar_previous) / denominator;
    result.mean_coefficient_xt =
        (1.0 - alpha_bar_previous) * std::sqrt(alpha) / denominator;

    if (!std::isfinite(result.posterior_variance) ||
        !std::isfinite(result.mean_coefficient_x0) ||
        !std::isfinite(result.mean_coefficient_xt)) {
        throw std::overflow_error("Non-finite reverse coefficients");
    }
    if (result.posterior_variance < 0.0) {
        throw std::overflow_error("Negative posterior variance");
    }
    return result;
}

double GaussianDiffusion::posterior_variance(
    std::size_t timestep
) const
{
    return reverse_coefficients(timestep).posterior_variance;
}

latent::core::Tensor GaussianDiffusion::predict_x0(
    const latent::core::Tensor& zt,
    std::size_t timestep,
    const latent::core::Tensor& predicted_epsilon
) const
{
    validate_reverse_inputs(zt, timestep, predicted_epsilon);

    const double signal_scale = schedule_.sqrt_alpha_bar(timestep);
    const double noise_scale =
        schedule_.sqrt_one_minus_alpha_bar(timestep);

    if (!(signal_scale > 0.0)) {
        throw std::overflow_error(
            "Invalid clean-latent reconstruction scale"
        );
    }

    latent::core::Tensor result(zt.shape());
    for (std::size_t i = 0; i < zt.size(); ++i) {
        const double value =
            (zt[i] - noise_scale * predicted_epsilon[i]) / signal_scale;
        if (!std::isfinite(value)) {
            throw std::overflow_error(
                "Predicted clean latent is non-finite"
            );
        }
        result[i] = value;
    }
    return result;
}

latent::core::Tensor GaussianDiffusion::reverse_mean(
    const latent::core::Tensor& zt,
    std::size_t timestep,
    const latent::core::Tensor& predicted_epsilon
) const
{
    validate_reverse_inputs(zt, timestep, predicted_epsilon);
    const ReverseCoefficients coefficients =
        reverse_coefficients(timestep);
    const latent::core::Tensor x0 =
        predict_x0(zt, timestep, predicted_epsilon);

    latent::core::Tensor result(zt.shape());
    for (std::size_t i = 0; i < zt.size(); ++i) {
        const double value =
            coefficients.mean_coefficient_x0 * x0[i] +
            coefficients.mean_coefficient_xt * zt[i];
        if (!std::isfinite(value)) {
            throw std::overflow_error(
                "Reverse mean produced non-finite output"
            );
        }
        result[i] = value;
    }
    return result;
}

} // namespace latent::diffusion
