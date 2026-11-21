
#include "latent/nn/training/MetricGradientController.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace latent::nn::training {
namespace {

using latent::core::Tensor;

static_assert(
    std::is_nothrow_move_assignable<Tensor>::value,
    "Atomic gradient commit requires noexcept Tensor move"
);

class StableNorm {
public:
    void add(double value)
    {
        const double x = std::abs(value);

        if (!std::isfinite(x)) {
            throw std::invalid_argument(
                "Non-finite gradient contribution"
            );
        }

        if (x == 0.0) return;

        if (scale_ < x) {
            const double ratio = scale_ / x;
            sumsq_ = 1.0 + sumsq_ * ratio * ratio;
            scale_ = x;
        } else {
            const double ratio = x / scale_;
            sumsq_ += ratio * ratio;
        }
    }

    double value() const
    {
        if (scale_ == 0.0) return 0.0;

        const double root = std::sqrt(sumsq_);

        if (scale_ >
            std::numeric_limits<double>::max() / root) {
            throw std::overflow_error(
                "Metric norm exceeds FP64"
            );
        }

        return scale_ * root;
    }

private:
    double scale_ = 0.0;
    double sumsq_ = 1.0;
};

} // namespace

MetricGradientController::MetricGradientController(
    std::vector<Parameter*> parameters,
    double max_norm
)
    : parameters_(std::move(parameters)),
      max_norm_(max_norm)
{
    if (parameters_.empty() ||
        !std::isfinite(max_norm_) ||
        max_norm_ <= 0.0) {
        throw std::invalid_argument(
            "Invalid metric controller configuration"
        );
    }

    std::unordered_set<Parameter*> seen;

    for (Parameter* parameter : parameters_) {
        if (!parameter ||
            !seen.insert(parameter).second) {
            throw std::invalid_argument(
                "Null or duplicate metric block"
            );
        }

        const auto& values = parameter->values();
        const auto& gradients = parameter->gradients();

        if (!values.same_shape(gradients) ||
            values.size() == 0) {
            throw std::invalid_argument(
                "Invalid metric block shape"
            );
        }

        registered_shapes_.push_back(values.shape());
    }
}

MetricGradientReport MetricGradientController::measure()
    const
{
    MetricGradientReport report;
    report.parameter_blocks = parameters_.size();

    StableNorm global;

    const double block_count =
        static_cast<double>(parameters_.size());

    for (std::size_t b = 0;
         b < parameters_.size(); ++b) {

        const auto& values = parameters_[b]->values();
        const auto& gradients =
            parameters_[b]->gradients();

        if (values.shape() != registered_shapes_[b] ||
            gradients.shape() != registered_shapes_[b]) {
            throw std::invalid_argument(
                "Registered metric block changed shape"
            );
        }

        if (gradients.size() >
            std::numeric_limits<std::size_t>::max() -
                report.parameter_count) {
            throw std::overflow_error(
                "Parameter count overflow"
            );
        }

        report.parameter_count += gradients.size();

        // sqrt(B) * sqrt(d_b) is the denominator
        // implied by M_b = I / (B * d_b).
        // Separating the factors avoids overflowing
        // the integer product B * d_b.
        const double divisor =
            std::sqrt(block_count) *
            std::sqrt(
                static_cast<double>(gradients.size())
            );

        for (std::size_t i = 0;
             i < gradients.size(); ++i) {

            global.add(gradients[i] / divisor);
        }
    }

    report.norm_before = global.value();
    report.norm_after = report.norm_before;

    return report;
}

MetricGradientReport MetricGradientController::apply()
{
    MetricGradientReport report = measure();

    if (report.norm_before <= max_norm_) {
        return report;
    }

    const double scale =
        max_norm_ / report.norm_before;

    std::vector<Tensor> staged;
    staged.reserve(parameters_.size());

    for (Parameter* parameter : parameters_) {
        Tensor replacement = parameter->gradients();

        for (std::size_t i = 0;
             i < replacement.size(); ++i) {

            replacement[i] *= scale;
        }

        if (!replacement.all_finite()) {
            throw std::overflow_error(
                "Non-finite clipped gradient"
            );
        }

        staged.push_back(std::move(replacement));
    }

    // All replacement gradients are now staged.
    // The final implementation should measure their
    // true metric norm before committing them.

    for (std::size_t b = 0;
         b < parameters_.size(); ++b) {

        parameters_[b]->gradients() =
            std::move(staged[b]);
    }

    report.applied_scale = scale;
    report.clipped = true;

    // Provisional expected value. Replace with the
    // measured norm of staged gradients before commit.
    report.norm_after = max_norm_;

    return report;
}

} // namespace latent::nn::training
