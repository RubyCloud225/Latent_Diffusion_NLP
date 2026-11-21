#include "latent/nn/training/TrainingGradientControl.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace latent::nn::training {
namespace {
using latent::core::Tensor;
using latent::nn::Parameter;

static_assert(std::is_nothrow_move_assignable<Tensor>::value,
              "Gradient rollback needs noexcept Tensor move assignment");

TrainingGradientMetrics as_metrics(
    GradientMode mode, const GradientReport& report)
{
    TrainingGradientMetrics out;
    out.mode = mode;
    out.norm_kind = "euclidean_l2";
    out.norm_before = report.norm_before;
    out.norm_after = report.norm_after;
    out.applied_scale = report.applied_scale;
    out.clipped = report.clipped;
    out.parameter_blocks = report.parameter_blocks;
    out.parameter_count = report.parameter_count;
    return out;
}

TrainingGradientMetrics as_metrics(const MetricGradientReport& report)
{
    TrainingGradientMetrics out;
    out.mode = GradientMode::MetricRadial;
    out.norm_kind = report.metric_id;
    out.metric_id = report.metric_id;
    out.norm_before = report.norm_before;
    out.norm_after = report.norm_after;
    out.applied_scale = report.applied_scale;
    out.clipped = report.clipped;
    out.parameter_blocks = report.parameter_blocks;
    out.parameter_count = report.parameter_count;
    return out;
}
} // namespace

TrainingGradientControl::TrainingGradientControl(
    std::vector<Parameter*> parameters,
    GradientMode mode,
    double max_norm,
    std::string metric_id)
    : parameters_(std::move(parameters)),
      mode_(mode),
      max_norm_(max_norm),
      metric_id_(std::move(metric_id))
{
    if (parameters_.empty() || !std::isfinite(max_norm_) ||
        max_norm_ <= 0.0) {
        throw std::invalid_argument("Invalid training gradient configuration");
    }
    switch (mode_) {
        case GradientMode::None:
        case GradientMode::EuclideanGlobal:
            if (!metric_id_.empty()) {
                throw std::invalid_argument("Metric ID requires metric mode");
            }
            break;
        case GradientMode::MetricRadial:
            if (metric_id_ != "equal_block_rms_v1") {
                throw std::invalid_argument("Unsupported metric ID");
            }
            break;
        default:
            throw std::invalid_argument("Unsupported gradient mode");
    }

    std::unordered_set<Parameter*> seen;
    for (Parameter* parameter : parameters_) {
        if (!parameter || !seen.insert(parameter).second ||
            !parameter->values().same_shape(parameter->gradients())) {
            throw std::invalid_argument("Invalid gradient parameter registration");
        }
        shapes_.push_back(parameter->values().shape());
    }
    validate_registration();
    if (mode_ == GradientMode::MetricRadial) {
        metric_controller_.reset(
            new MetricGradientController(parameters_, max_norm_));
    }
}

void TrainingGradientControl::validate_registration() const
{
    for (std::size_t i = 0; i < parameters_.size(); ++i) {
        if (parameters_[i]->values().shape() != shapes_[i] ||
            parameters_[i]->gradients().shape() != shapes_[i] ||
            !parameters_[i]->values().all_finite() ||
            !parameters_[i]->gradients().all_finite()) {
            throw std::invalid_argument(
                "Registered gradient parameter changed or became non-finite");
        }
    }
}

TrainingGradientMetrics TrainingGradientControl::measure() const
{
    validate_registration();
    if (mode_ == GradientMode::MetricRadial) {
        return as_metrics(metric_controller_->measure());
    }
    const GradientController controller(max_norm_);
    return as_metrics(mode_, controller.measure(parameters_));
}

void TrainingGradientControl::record_rejection() noexcept
{
    if (counters_.rejected_applications <
        std::numeric_limits<std::size_t>::max()) {
        ++counters_.rejected_applications;
    }
}

TrainingGradientMetrics TrainingGradientControl::apply()
{
    std::vector<Tensor> original_gradients;
    try {
        validate_registration();
        if (counters_.successful_applications ==
            std::numeric_limits<std::size_t>::max() ||
            counters_.clipped_applications ==
            std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error("Gradient controller counters exhausted");
        }
        original_gradients.reserve(parameters_.size());
        for (Parameter* parameter : parameters_) {
            original_gradients.push_back(parameter->gradients());
        }
    } catch (...) {
        record_rejection();
        throw;
    }

    try {
        TrainingGradientMetrics result;
        if (mode_ == GradientMode::MetricRadial) {
            result = as_metrics(metric_controller_->apply());
        } else if (mode_ == GradientMode::EuclideanGlobal) {
            GradientController controller(max_norm_);
            result = as_metrics(mode_, controller.apply(parameters_));
        } else {
            result = measure();
        }

        // Re-read the final gradient tensors. In Euclidean mode the
        // underlying NN.1.07C report uses the desired threshold,
        // rather than the measured post-rounding result.
        // The metric controller also uses its own staged measurement.
        validate_registration();
        const auto final_measurement = measure();
        result.norm_after = final_measurement.norm_before;
        if (!std::isfinite(result.norm_before) ||
            !std::isfinite(result.norm_after) ||
            !std::isfinite(result.applied_scale) ||
            result.applied_scale <= 0.0 ||
            result.norm_after > result.norm_before) {
            throw std::overflow_error("Invalid post-clipping gradient metrics");
        }

        ++counters_.successful_applications;
        if (result.clipped) ++counters_.clipped_applications;
        return result;
    } catch (...) {
        // All original tensors were staged, and move assignment
        // is statically verified noexcept before any mutation.
        for (std::size_t i = 0; i < parameters_.size(); ++i) {
            parameters_[i]->gradients() = std::move(original_gradients[i]);
        }
        record_rejection();
        throw;
    }
}

GradientControlCounters TrainingGradientControl::counters() const noexcept
{
    return counters_;
}

} // namespace latent::nn::training
