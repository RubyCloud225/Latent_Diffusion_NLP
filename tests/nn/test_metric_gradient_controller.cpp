
#include "latent/nn/training/MetricGradientController.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <stdexcept>

using latent::nn::Parameter;
using latent::nn::training::MetricGradientController;

namespace {

void near(double actual, double expected)
{
    if (!std::isfinite(actual) ||
        std::abs(actual - expected) > 1e-12) {
        throw std::runtime_error("Numerical mismatch");
    }
}

void test_known_answer()
{
    Parameter weights({2});
    Parameter bias({1});

    weights.gradients()[0] = 3.0;
    weights.gradients()[1] = 4.0;
    bias.gradients()[0] = 12.0;

    MetricGradientController controller(
        {&weights, &bias}, 1.0
    );

    const auto before = controller.measure();

    const double expected_norm =
        std::sqrt(78.25);

    near(before.norm_before, expected_norm);

    const auto after = controller.apply();

    const double expected_scale =
        1.0 / expected_norm;

    near(after.applied_scale, expected_scale);

    near(
        weights.gradients()[0],
        3.0 * expected_scale
    );

    near(
        weights.gradients()[1],
        4.0 * expected_scale
    );

    near(
        bias.gradients()[0],
        12.0 * expected_scale
    );

    near(controller.measure().norm_before, 1.0);
}

void test_zero_gradient()
{
    Parameter weights({4});

    MetricGradientController controller(
        {&weights}, 1.0
    );

    const auto result = controller.apply();

    near(result.norm_before, 0.0);
    near(result.applied_scale, 1.0);

    if (result.clipped) {
        throw std::runtime_error(
            "Zero gradients must remain unchanged"
        );
    }
}

void test_block_normalisation()
{
    Parameter small({1});
    Parameter large({4});

    small.gradients()[0] = 2.0;
    large.gradients().fill(2.0);

    MetricGradientController controller(
        {&small, &large}, 10.0
    );

    // Both blocks have RMS gradient 2.0.
    // Equal-block metric norm must therefore be 2.0.
    near(controller.measure().norm_before, 2.0);
}

void test_invalid_registration()
{
    Parameter parameter({2});

    bool rejected = false;

    try {
        MetricGradientController invalid(
            {&parameter, &parameter}, 1.0
        );
    } catch (const std::invalid_argument&) {
        rejected = true;
    }

    if (!rejected) {
        throw std::runtime_error(
            "Duplicate parameter not rejected"
        );
    }
}


template<typename Exception, typename Function>
void rejects(Function function)
{
    bool caught = false;
    try {
        function();
    } catch (const Exception&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("Expected rejection missing");
    }
}

void test_report_matches_measured_norm()
{
    Parameter first({2});
    Parameter second({1});

    first.gradients()[0] = 3.0;
    first.gradients()[1] = 4.0;
    second.gradients()[0] = 12.0;

    MetricGradientController controller(
        {&first, &second}, 0.7
    );

    const auto result = controller.apply();
    const auto measured = controller.measure();

    near(result.norm_after, measured.norm_before);
    near(result.applied_scale,
         0.7 / std::sqrt(78.25));

    if (!result.clipped ||
        result.norm_after > result.norm_before) {
        throw std::runtime_error("Invalid clipping diagnostics");
    }
}

void test_measure_does_not_mutate()
{
    Parameter first({2});
    first.gradients()[0] = 3.0;
    first.gradients()[1] = 4.0;
    first.values().fill(17.0);

    MetricGradientController controller({&first}, 1.0);

    const auto initial = controller.measure();
    const auto repeated = controller.measure();

    near(initial.norm_before, repeated.norm_before);
    near(first.gradients()[0], 3.0);
    near(first.gradients()[1], 4.0);
    near(first.values()[0], 17.0);
    near(first.values()[1], 17.0);
}

void test_noop_below_threshold()
{
    Parameter first({2});
    first.gradients()[0] = 0.3;
    first.gradients()[1] = 0.4;

    MetricGradientController controller({&first}, 1.0);
    const auto result = controller.apply();

    if (result.clipped || result.applied_scale != 1.0) {
        throw std::runtime_error("Unexpected clipping");
    }

    near(first.gradients()[0], 0.3);
    near(first.gradients()[1], 0.4);
    near(result.norm_before, result.norm_after);
}

void test_invalid_gradients_atomic()
{
    Parameter valid({1});
    Parameter invalid({1});
    valid.gradients()[0] = 9.0;
    invalid.gradients()[0] =
        std::numeric_limits<double>::quiet_NaN();

    MetricGradientController controller(
        {&valid, &invalid}, 1.0
    );

    rejects<std::invalid_argument>([&] {
        controller.apply();
    });

    near(valid.gradients()[0], 9.0);
    if (!std::isnan(invalid.gradients()[0])) {
        throw std::runtime_error(
            "Invalid gradient was changed"
        );
    }
}

void test_weighted_underflow_atomic()
{
    Parameter first({4});
    first.gradients()[0] =
        std::numeric_limits<double>::denorm_min();
    first.gradients()[1] = 0.0;
    first.gradients()[2] = 0.0;
    first.gradients()[3] = 0.0;

    MetricGradientController controller({&first}, 1.0);

    rejects<std::underflow_error>([&] {
        controller.measure();
    });

    rejects<std::underflow_error>([&] {
        controller.apply();
    });

    if (first.gradients()[0] !=
        std::numeric_limits<double>::denorm_min()) {
        throw std::runtime_error(
            "Weighted underflow modified gradient"
        );
    }
}

void test_clipping_underflow_atomic()
{
    Parameter first({2});
    first.gradients()[0] = 1e200;
    first.gradients()[1] = 1e-200;

    MetricGradientController controller({&first}, 1.0);

    rejects<std::underflow_error>([&] {
        controller.apply();
    });

    near(first.gradients()[0], 1e200);
    near(first.gradients()[1], 1e-200);
}

void test_large_finite_gradient()
{
    Parameter first({2});
    first.gradients()[0] = 1e200;
    first.gradients()[1] = 1e200;

    MetricGradientController controller({&first}, 1.0);
    const auto report = controller.apply();

    if (!std::isfinite(report.norm_before) ||
        !std::isfinite(report.norm_after)) {
        throw std::runtime_error("Non-finite metric norm");
    }

    near(report.norm_after,
         controller.measure().norm_before);
    near(first.gradients()[0], 1.0);
    near(first.gradients()[1], 1.0);
}

void test_overflow_atomic()
{
    Parameter first({2});
    const double maximum =
        std::numeric_limits<double>::max();
    first.gradients()[0] = maximum;
    first.gradients()[1] = maximum;

    MetricGradientController controller({&first}, 1.0);

    rejects<std::overflow_error>([&] {
        controller.apply();
    });

    if (first.gradients()[0] != maximum ||
        first.gradients()[1] != maximum) {
        throw std::runtime_error(
            "Overflow changed gradients"
        );
    }
}

void test_invalid_configuration()
{
    Parameter first({2});

    rejects<std::invalid_argument>([&] {
        MetricGradientController controller({}, 1.0);
    });

    rejects<std::invalid_argument>([&] {
        MetricGradientController controller({nullptr}, 1.0);
    });

    rejects<std::invalid_argument>([&] {
        MetricGradientController controller({&first}, 0.0);
    });

    rejects<std::invalid_argument>([&] {
        MetricGradientController controller(
            {&first},
            std::numeric_limits<double>::infinity()
        );
    });
}

} // namespace

int main()
{
    try {
        test_known_answer();
        test_zero_gradient();
        test_block_normalisation();
        test_invalid_registration();
        test_report_matches_measured_norm();
        test_measure_does_not_mutate();
        test_noop_below_threshold();
        test_invalid_gradients_atomic();
        test_weighted_underflow_atomic();
        test_clipping_underflow_atomic();
        test_large_finite_gradient();
        test_overflow_atomic();
        test_invalid_configuration();

        std::cout
            << "[PASS] NN.1.07D Metric Gradient Controller\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "[FAIL] NN.1.07D: "
            << error.what() << '\n';

        return 1;
    }
}
