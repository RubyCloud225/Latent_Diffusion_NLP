
#include "latent/nn/AdamFP64.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace latent::nn {

namespace {

/**
 * @brief Numerically stable Adam bias correction.
 *
 * Computes 1 - beta^step without cancellation when
 * beta is close to one.
 */
double bias_correction(
    double beta,
    std::size_t step
)
{
    if (beta == 0.0) {
        return 1.0;
    }

    return -std::expm1(
        static_cast<double>(step) * std::log(beta)
    );
}

void require_finite(
    const latent::core::Tensor& tensor
)
{
    if (!tensor.all_finite()) {
        throw std::invalid_argument(
            "AdamFP64 requires finite Tensor values"
        );
    }
}

} // namespace

AdamFP64::AdamFP64(
    std::vector<Parameter*> parameters,
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon
)
    : parameters_(std::move(parameters)),
      learning_rate_(learning_rate),
      beta1_(beta1),
      beta2_(beta2),
      epsilon_(epsilon)
{
    if (!std::isfinite(learning_rate_) ||
        learning_rate_ <= 0.0 ||
        !std::isfinite(beta1_) ||
        beta1_ < 0.0 || beta1_ >= 1.0 ||
        !std::isfinite(beta2_) ||
        beta2_ < 0.0 || beta2_ >= 1.0 ||
        !std::isfinite(epsilon_) ||
        epsilon_ <= 0.0) {
        throw std::invalid_argument(
            "AdamFP64 invalid hyperparameters"
        );
    }

    if (parameters_.empty()) {
        throw std::invalid_argument(
            "AdamFP64 requires at least one parameter"
        );
    }

    std::unordered_set<Parameter*> seen;

    states_.reserve(parameters_.size());

    for (Parameter* parameter : parameters_) {
        if (parameter == nullptr ||
            !seen.insert(parameter).second) {
            throw std::invalid_argument(
                "AdamFP64 requires unique non-null parameters"
            );
        }

        if (!parameter->values().same_shape(
                parameter->gradients())) {
            throw std::invalid_argument(
                "AdamFP64 parameter shape mismatch"
            );
        }

        states_.emplace_back(
            parameter->values().shape()
        );
    }
}

void AdamFP64::step()
{
    if (step_ ==
        std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error(
            "AdamFP64 step counter overflow"
        );
    }

    const std::size_t next_step = step_ + 1;

    const double correction1 =
        bias_correction(beta1_, next_step);

    const double correction2 =
        bias_correction(beta2_, next_step);

    if (!std::isfinite(correction1) ||
        !std::isfinite(correction2) ||
        correction1 <= 0.0 ||
        correction2 <= 0.0) {
        throw std::overflow_error(
            "AdamFP64 invalid bias correction"
        );
    }

    // Prepare complete replacement tensors.
    // Nothing observable is mutated during this phase.
    std::vector<latent::core::Tensor> new_values;
    std::vector<State> new_states;

    new_values.reserve(parameters_.size());
    new_states.reserve(parameters_.size());

    for (std::size_t p = 0;
         p < parameters_.size(); ++p) {

        Parameter& parameter = *parameters_[p];
        const auto& values = parameter.values();
        const auto& gradients = parameter.gradients();
        const State& old = states_[p];

        if (!values.same_shape(gradients) ||
            !values.same_shape(old.first) ||
            !values.same_shape(old.second)) {
            throw std::invalid_argument(
                "AdamFP64 parameter shape changed"
            );
        }

        require_finite(values);
        require_finite(gradients);
        require_finite(old.first);
        require_finite(old.second);

        latent::core::Tensor updated = values;
        State next(values.shape());

        for (std::size_t i = 0;
             i < values.size(); ++i) {

            const double g = gradients[i];

            const double m =
                beta1_ * old.first[i] +
                (1.0 - beta1_) * g;

            const double v =
                beta2_ * old.second[i] +
                (1.0 - beta2_) * g * g;

            if (!std::isfinite(m) ||
                !std::isfinite(v) ||
                v < 0.0) {
                throw std::overflow_error(
                    "AdamFP64 moment overflow"
                );
            }

            const double corrected_m =
                m / correction1;

            const double corrected_v =
                v / correction2;

            const double denominator =
                std::sqrt(corrected_v) + epsilon_;

            const double delta =
                learning_rate_ *
                (corrected_m / denominator);

            const double value =
                values[i] - delta;

            if (!std::isfinite(corrected_m) ||
                !std::isfinite(corrected_v) ||
                !std::isfinite(denominator) ||
                denominator <= 0.0 ||
                !std::isfinite(delta) ||
                !std::isfinite(value)) {
                throw std::overflow_error(
                    "AdamFP64 update overflow"
                );
            }

            next.first[i] = m;
            next.second[i] = v;
            updated[i] = value;
        }

        new_values.push_back(std::move(updated));
        new_states.push_back(std::move(next));
    }

    // All proposed updates have passed validation.
    //
    // Tensor uses owning vector-backed storage. The
    // commit below uses move assignment into existing
    // tensors, retaining Parameter object identities.
    for (std::size_t p = 0;
         p < parameters_.size(); ++p) {

        parameters_[p]->values() =
            std::move(new_values[p]);
    }

    states_ = std::move(new_states);
    step_ = next_step;
}

void AdamFP64::zero_grad()
{
    for (Parameter* parameter : parameters_) {
        parameter->zero_grad();
    }
}

void AdamFP64::reset()
{
    for (State& state : states_) {
        state.first.fill(0.0);
        state.second.fill(0.0);
    }

    step_ = 0;
}

std::size_t AdamFP64::step_count() const noexcept
{
    return step_;
}

} // namespace latent::nn
