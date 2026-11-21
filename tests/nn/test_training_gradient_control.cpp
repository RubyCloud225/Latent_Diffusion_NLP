#include "latent/nn/training/TrainingGradientControl.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using latent::nn::Parameter;
using latent::nn::training::GradientMode;
using latent::nn::training::TrainingGradientControl;

namespace {
void require(bool ok, const char* msg)
{
    if (!ok) throw std::runtime_error(msg);
}

void near(double actual, double expected, double tolerance = 1e-12)
{
    require(std::isfinite(actual) &&
            std::abs(actual - expected) <= tolerance,
            "Gradient control numerical mismatch");
}

template<class E, class F>
void rejects(F function)
{
    bool caught = false;
    try { function(); } catch (const E&) { caught = true; }
    require(caught, "Expected gradient control rejection missing");
}

void test_none_readonly_and_counters()
{
    Parameter parameter({2});
    parameter.gradients()[0] = 3.0;
    parameter.gradients()[1] = 4.0;
    TrainingGradientControl control(
        {&parameter}, GradientMode::None, 1.0);

    const auto measured = control.measure();
    const auto before = control.counters();
    near(measured.norm_before, 5.0);
    near(measured.norm_after, 5.0);
    require(before.successful_applications == 0,
            "Read-only measurement changed counters");

    const auto applied = control.apply();
    near(applied.norm_before, 5.0);
    near(applied.norm_after, 5.0);
    near(applied.applied_scale, 1.0);
    require(!applied.clipped && applied.norm_kind == "euclidean_l2",
            "None mode unexpectedly clipped");
    near(parameter.gradients()[0], 3.0);
    near(parameter.gradients()[1], 4.0);
    const auto after = control.counters();
    require(after.successful_applications == 1 &&
            after.clipped_applications == 0 &&
            after.rejected_applications == 0,
            "None-mode counters incorrect");
}

void test_euclidean_actual_post_norm()
{
    Parameter parameter({2});
    parameter.gradients()[0] = 3.0;
    parameter.gradients()[1] = 4.0;
    TrainingGradientControl control(
        {&parameter}, GradientMode::EuclideanGlobal, 1.0);
    const auto applied = control.apply();
    require(applied.clipped && applied.parameter_count == 2 &&
            applied.parameter_blocks == 1,
            "Euclidean report invalid");
    near(applied.norm_before, 5.0);
    near(applied.applied_scale, 0.2);
    near(parameter.gradients()[0], 0.6);
    near(parameter.gradients()[1], 0.8);
    near(applied.norm_after, control.measure().norm_before, 0.0);
    require(control.counters().clipped_applications == 1,
            "Clipping counter missing");
}

void test_metric_block_norm_and_identity()
{
    Parameter weights({2});
    Parameter bias({1});
    weights.gradients()[0] = 3.0;
    weights.gradients()[1] = 4.0;
    bias.gradients()[0] = 12.0;
    TrainingGradientControl control(
        {&weights, &bias}, GradientMode::MetricRadial,
        1.0, "equal_block_rms_v1");
    const auto result = control.apply();
    near(result.norm_before, std::sqrt(78.25));
    near(result.applied_scale, 1.0 / std::sqrt(78.25));
    near(result.norm_after, control.measure().norm_before, 0.0);
    require(result.clipped &&
            result.norm_kind == "equal_block_rms_v1" &&
            result.metric_id == "equal_block_rms_v1",
            "Metric identity missing from diagnostics");
    require(control.counters().successful_applications == 1 &&
            control.counters().clipped_applications == 1,
            "Metric counters invalid");
}

void test_euclidean_and_metric_are_not_interchangeable()
{
    Parameter a({2});
    Parameter b({1});
    a.gradients()[0] = 3.0;
    a.gradients()[1] = 4.0;
    b.gradients()[0] = 12.0;

    TrainingGradientControl l2(
        {&a, &b}, GradientMode::EuclideanGlobal, 100.0);
    TrainingGradientControl metric(
        {&a, &b}, GradientMode::MetricRadial,
        100.0, "equal_block_rms_v1");
    near(l2.measure().norm_before, 13.0);
    near(metric.measure().norm_before, std::sqrt(78.25));
}

void test_invalid_registration_and_configuration()
{
    Parameter parameter({2});
    rejects<std::invalid_argument>([&] {
        TrainingGradientControl bad(
            {&parameter, &parameter}, GradientMode::None, 1.0);
    });
    rejects<std::invalid_argument>([&] {
        TrainingGradientControl bad(
            {nullptr}, GradientMode::None, 1.0);
    });
    rejects<std::invalid_argument>([&] {
        TrainingGradientControl bad(
            {}, GradientMode::None, 1.0);
    });
    rejects<std::invalid_argument>([&] {
        TrainingGradientControl bad(
            {&parameter}, GradientMode::EuclideanGlobal, 0.0);
    });
    rejects<std::invalid_argument>([&] {
        TrainingGradientControl bad(
            {&parameter}, GradientMode::MetricRadial,
            1.0, "invalid_metric");
    });
    rejects<std::invalid_argument>([&] {
        TrainingGradientControl bad(
            {&parameter}, GradientMode::None,
            1.0, "equal_block_rms_v1");
    });
    rejects<std::invalid_argument>([&] {
        TrainingGradientControl bad(
            {&parameter}, static_cast<GradientMode>(123), 1.0);
    });
}

void test_failure_preserves_gradients_and_counters()
{
    Parameter parameter({2});
    parameter.gradients()[0] = 3.0;
    parameter.gradients()[1] = 4.0;
    TrainingGradientControl control(
        {&parameter}, GradientMode::EuclideanGlobal, 1.0);
    parameter.gradients()[1] =
        std::numeric_limits<double>::quiet_NaN();
    rejects<std::invalid_argument>([&] {
        (void)control.apply();
    });
    near(parameter.gradients()[0], 3.0);
    require(std::isnan(parameter.gradients()[1]),
            "Invalid gradient silently overwritten");
    const auto stats = control.counters();
    require(stats.successful_applications == 0 &&
            stats.clipped_applications == 0 &&
            stats.rejected_applications == 1,
            "Rejected application miscounted");
}

void test_metric_destructive_underflow_rollback()
{
    Parameter parameter({4});
    parameter.gradients()[0] =
        std::numeric_limits<double>::denorm_min();
    TrainingGradientControl control(
        {&parameter}, GradientMode::MetricRadial,
        1.0, "equal_block_rms_v1");
    rejects<std::underflow_error>([&] {
        (void)control.apply();
    });
    require(parameter.gradients()[0] ==
            std::numeric_limits<double>::denorm_min(),
            "Underflow failure changed parameter gradient");
    require(control.counters().rejected_applications == 1 &&
            control.counters().successful_applications == 0,
            "Underflow incorrectly recorded as success");
}

} // namespace

int main()
{
    try {
        test_none_readonly_and_counters();
        test_euclidean_actual_post_norm();
        test_metric_block_norm_and_identity();
        test_euclidean_and_metric_are_not_interchangeable();
        test_invalid_registration_and_configuration();
        test_failure_preserves_gradients_and_counters();
        test_metric_destructive_underflow_rollback();
        std::cout << "[PASS] NN.1.07E.05 training gradient control\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] NN.1.07E.05: "
                  << error.what() << '\n';
        return 1;
    }
}
