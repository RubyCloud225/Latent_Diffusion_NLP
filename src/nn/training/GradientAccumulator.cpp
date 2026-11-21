#include "latent/nn/training/GradientAccumulator.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace latent::nn::training {
namespace {
using latent::core::Tensor;
using latent::nn::Parameter;

static_assert(std::is_nothrow_move_assignable<Tensor>::value,
              "Rollback requires noexcept Tensor move assignment");
static_assert(std::is_nothrow_move_assignable<ImageBatchSource>::value,
              "Rollback requires noexcept ImageBatchSource move assignment");

bool same_tensor(const Tensor& a, const Tensor& b)
{
    if (!a.same_shape(b)) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

void require_exact_count(std::size_t count)
{
    // Above 2^53, not all integer counts are exactly representable
    // in an IEEE FP64 loss_scale or sample denominator.
    if (count == 0 || static_cast<std::uint64_t>(count) >
                          UINT64_C(9007199254740992)) {
        throw std::overflow_error("Image sample count not representable exactly in FP64");
    }
}
} // namespace

GradientAccumulator::GradientAccumulator(
    ImageTrainingTask& task,
    ImageBatchSource& source,
    std::size_t accumulation_steps)
    : task_(task), source_(source), accumulation_steps_(accumulation_steps)
{
    if (accumulation_steps_ == 0) {
        throw std::invalid_argument("Accumulation steps must be positive");
    }
}

AccumulationReport GradientAccumulator::accumulate_window()
{
    // Copy before any changes: this copy contains the entire deterministic
    // epoch order and cursor, and its noexcept move restores both.
    ImageBatchSource source_backup = source_;
    auto registered = task_.parameters();
    if (registered.empty()) {
        throw std::invalid_argument("Accumulator requires registered parameters");
    }

    std::unordered_set<Parameter*> seen;
    std::vector<Tensor> old_values;
    std::vector<Tensor> old_gradients;
    old_values.reserve(registered.size());
    old_gradients.reserve(registered.size());

    for (Parameter* parameter : registered) {
        if (!parameter || !seen.insert(parameter).second) {
            throw std::invalid_argument("Null or duplicate accumulation parameter");
        }
        if (!parameter->values().same_shape(parameter->gradients()) ||
            !parameter->values().all_finite() ||
            !parameter->gradients().all_finite()) {
            throw std::invalid_argument("Invalid registered image parameter");
        }
        for (std::size_t i = 0; i < parameter->gradients().size(); ++i) {
            if (parameter->gradients()[i] != 0.0) {
                throw std::invalid_argument(
                    "Accumulation requires zero gradients at window start");
            }
        }
        old_values.push_back(parameter->values());
        old_gradients.push_back(parameter->gradients());
    }

    AccumulationReport report;
    double weighted_loss = 0.0;
    bool saw_partial_batch = false;

    try {
        for (std::size_t j = 0; j < accumulation_steps_; ++j) {
            auto batch = source_.next();
            if (!batch.has_value()) {
                report.end_of_epoch = true;
                break;
            }
            batch->validate();
            const std::size_t n = batch->sample_count();
            require_exact_count(n);
            if (n > std::numeric_limits<std::size_t>::max() - report.samples) {
                throw std::overflow_error("Accumulated image count overflow");
            }
            const std::size_t next_total = report.samples + n;
            require_exact_count(next_total);
            if (n < source_.batch_size()) saw_partial_batch = true;

            // TrainingTask accumulates n * d(mean microbatch loss)/d(theta).
            const double batch_loss = task_.train_microbatch(
                *batch, static_cast<double>(n));
            if (!std::isfinite(batch_loss) || batch_loss < 0.0) {
                throw std::invalid_argument("Invalid image microbatch mean loss");
            }
            const double contribution = static_cast<double>(n) * batch_loss;
            if (!std::isfinite(contribution) ||
                !std::isfinite(weighted_loss + contribution)) {
                throw std::overflow_error("Sample-weighted image loss overflow");
            }
            weighted_loss += contribution;
            report.samples = next_total;
            ++report.microbatches;
        }

        if (report.samples == 0) {
            return report; // No task call or mutation at end of epoch.
        }

        const double denominator = static_cast<double>(report.samples);
        report.mean_loss = weighted_loss / denominator;
        if (!std::isfinite(report.mean_loss)) {
            throw std::overflow_error("Non-finite image accumulation loss");
        }
        report.partial_window =
            saw_partial_batch || report.microbatches < accumulation_steps_;

        std::vector<Tensor> staged;
        staged.reserve(registered.size());
        for (std::size_t p = 0; p < registered.size(); ++p) {
            if (!same_tensor(registered[p]->values(), old_values[p]) ||
                !registered[p]->values().same_shape(
                    registered[p]->gradients())) {
                throw std::invalid_argument(
                    "Task modified parameters or gradient layout during accumulation");
            }
            Tensor normalised = registered[p]->gradients();
            if (!normalised.all_finite()) {
                throw std::invalid_argument("Non-finite accumulated gradient");
            }
            for (std::size_t i = 0; i < normalised.size(); ++i) {
                normalised[i] /= denominator;
            }
            if (!normalised.all_finite()) {
                throw std::overflow_error("Non-finite normalised gradient");
            }
            staged.push_back(std::move(normalised));
        }

        // No throws after this point. Values and source cursor are unchanged
        // except for the successfully consumed image microbatches.
        for (std::size_t p = 0; p < registered.size(); ++p) {
            registered[p]->gradients() = std::move(staged[p]);
        }
        task_.reset_transient_state();
        return report;
    } catch (...) {
        for (std::size_t p = 0; p < registered.size(); ++p) {
            registered[p]->values() = std::move(old_values[p]);
            registered[p]->gradients() = std::move(old_gradients[p]);
        }
        source_ = std::move(source_backup);
        task_.reset_transient_state();
        throw;
    }
}

} // namespace latent::nn::training
