
#include "latent/nn/training/GradientController.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace latent::nn::training {
namespace {

struct StableNorm {
    double scale = 0.0;
    double sumsq = 1.0;

    void add(double value)
    {
        const double a = std::abs(value);

        if (!std::isfinite(a)) {
            throw std::invalid_argument(
                "Non-finite gradient"
            );
        }

        if (a == 0.0) return;

        if (scale < a) {
            const double ratio = scale / a;
            sumsq = 1.0 + sumsq * ratio * ratio;
            scale = a;
        } else {
            const double ratio = a / scale;
            sumsq += ratio * ratio;
        }
    }

    double value() const
    {
        if (scale == 0.0) return 0.0;

        const double root = std::sqrt(sumsq);

        if (scale >
            std::numeric_limits<double>::max() / root) {
            throw std::overflow_error(
                "Global L2 norm exceeds FP64 range"
            );
        }

        return scale * root;
    }
};

struct Measurement {
    double norm = 0.0;
    std::size_t elements = 0;
};

Measurement inspect(
    const std::vector<Parameter*>& parameters
)
{
    if (parameters.empty()) {
        throw std::invalid_argument(
            "No parameters registered"
        );
    }

    std::unordered_set<Parameter*> seen;
    StableNorm accumulator;
    std::size_t total_elements = 0;

    for (Parameter* parameter : parameters) {
        if (!parameter ||
            !seen.insert(parameter).second) {
            throw std::invalid_argument(
                "Null or duplicate parameter"
            );
        }

        const auto& gradient = parameter->gradients();
        const auto& values = parameter->values();

        if (!gradient.same_shape(values)) {
            throw std::invalid_argument(
                "Gradient shape mismatch"
            );
        }

        if (gradient.size() >
            std::numeric_limits<std::size_t>::max() -
                total_elements) {
            throw std::overflow_error(
                "Gradient element-count overflow"
            );
        }

        total_elements += gradient.size();

        for (std::size_t i = 0;
             i < gradient.size(); ++i) {
            accumulator.add(gradient[i]);
        }
    }

    return {accumulator.value(), total_elements};
}

GradientReport report_for(
    double norm,
    std::size_t blocks,
    std::size_t elements
)
{
    GradientReport report;
    report.norm_before = norm;
    report.norm_after = norm;
    report.parameter_blocks = blocks;
    report.parameter_count = elements;
    return report;
}

} // namespace

GradientController::GradientController(double max_norm)
    : max_norm_(max_norm)
{
    if (!std::isfinite(max_norm_) || max_norm_ <= 0.0) {
        throw std::invalid_argument(
            "Gradient limit must be finite and positive"
        );
    }
}

GradientReport GradientController::measure(
    const std::vector<Parameter*>& parameters
) const
{
    const auto result = inspect(parameters);

    return report_for(
        result.norm,
        parameters.size(),
        result.elements
    );
}

GradientReport GradientController::apply(
    const std::vector<Parameter*>& parameters
) const
{
    GradientReport report = measure(parameters);

    if (report.norm_before <= max_norm_) {
        return report;
    }

    const double factor = max_norm_ / report.norm_before;

    // Allocate and validate all replacement gradients
    // before changing any registered Parameter.
    std::vector<latent::core::Tensor> staged;
    staged.reserve(parameters.size());

    for (Parameter* parameter : parameters) {
        latent::core::Tensor gradient =
            parameter->gradients();

        for (std::size_t i = 0;
             i < gradient.size(); ++i) {
            gradient[i] *= factor;
        }

        if (!gradient.all_finite()) {
            throw std::overflow_error(
                "Clipped gradient is non-finite"
            );
        }

        staged.push_back(std::move(gradient));
    }

    // Commit once all replacement tensors are ready.
    // Tensor's owning vector move assignment is
    // expected to be noexcept for this storage type;
    // that property should be checked explicitly.
    for (std::size_t i = 0; i < parameters.size(); ++i) {
        parameters[i]->gradients() = std::move(staged[i]);
    }

    report.applied_scale = factor;
    report.clipped = true;

    // Report the intended clipped norm. For additional
    // diagnostic fidelity, compute the actual resulting
    // norm after the staged arithmetic in NN.1.07C.
    report.norm_after = max_norm_;

    return report;
}

} // namespace latent::nn::training
