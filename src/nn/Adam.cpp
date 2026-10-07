#include "../../latent/nn/Adam.hpp"

#include <cmath>
#include <stdexcept>

namespace latent::nn {

Adam::Adam(
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon
)
    : learning_rate_(learning_rate),
      beta1_(beta1),
      beta2_(beta2),
      epsilon_(epsilon)
{
    if (learning_rate <= 0.0f) {
        throw std::invalid_argument(
            "Adam learning rate must be positive."
        );
    }

    if (beta1 < 0.0f || beta1 >= 1.0f) {
        throw std::invalid_argument(
            "Adam beta1 must be in [0, 1)."
        );
    }

    if (beta2 < 0.0f || beta2 >= 1.0f) {
        throw std::invalid_argument(
            "Adam beta2 must be in [0, 1)."
        );
    }

    if (epsilon <= 0.0f) {
        throw std::invalid_argument(
            "Adam epsilon must be positive."
        );
    }
}


void Adam::step(
    std::vector<float>& parameters,
    const std::vector<float>& gradients
)
{
    if (parameters.size() != gradients.size()) {
        throw std::invalid_argument(
            "Adam parameter and gradient sizes must match."
        );
    }

    if (first_moment_.empty()) {
        first_moment_.assign(
            parameters.size(),
            0.0f
        );

        second_moment_.assign(
            parameters.size(),
            0.0f
        );
    }

    if (
        first_moment_.size() != parameters.size() ||
        second_moment_.size() != parameters.size()
    ) {
        throw std::invalid_argument(
            "Adam parameter count changed after initialisation."
        );
    }

    ++step_;

    const float beta1_correction =
        1.0f -
        std::pow(
            beta1_,
            static_cast<float>(step_)
        );

    const float beta2_correction =
        1.0f -
        std::pow(
            beta2_,
            static_cast<float>(step_)
        );

    for (
        std::size_t i = 0;
        i < parameters.size();
        ++i
    ) {
        const float gradient =
            gradients[i];

        first_moment_[i] =
            beta1_ * first_moment_[i] +
            (1.0f - beta1_) * gradient;

        second_moment_[i] =
            beta2_ * second_moment_[i] +
            (1.0f - beta2_) *
                gradient * gradient;

        const float corrected_first =
            first_moment_[i] /
            beta1_correction;

        const float corrected_second =
            second_moment_[i] /
            beta2_correction;

        parameters[i] -=
            learning_rate_ *
            corrected_first /
            (
                std::sqrt(
                    corrected_second
                ) +
                epsilon_
            );
    }
}


void Adam::reset()
{
    step_ = 0;
    first_moment_.clear();
    second_moment_.clear();
}


std::size_t
Adam::step_count() const noexcept
{
    return step_;
}

} // namespace latent::nn