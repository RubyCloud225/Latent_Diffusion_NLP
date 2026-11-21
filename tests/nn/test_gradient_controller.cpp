
#include "latent/nn/training/GradientController.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using latent::nn::Parameter;
using latent::nn::training::GradientController;

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(double actual, double expected, double tolerance)
{
    require(
        std::isfinite(actual) &&
        std::abs(actual - expected) <= tolerance,
        "Numerical mismatch"
    );
}

template<class Exception, class F>
void rejects(F operation)
{
    bool caught = false;

    try {
        operation();
    } catch (const Exception&) {
        caught = true;
    }

    require(caught, "Expected rejection missing");
}

void test_known_answer()
{
    Parameter weights({2});
    weights.gradients()[0] = 3.0;
    weights.gradients()[1] = 4.0;

    GradientController controller(1.0);

    const auto report = controller.apply({&weights});

    near(report.norm_before, 5.0, 1e-12);
    near(report.norm_after, 1.0, 1e-12);
    near(report.applied_scale, 0.2, 1e-12);

    near(weights.gradients()[0], 0.6, 1e-12);
    near(weights.gradients()[1], 0.8, 1e-12);

    require(report.clipped, "Clipping was not applied");
}

void test_global_across_parameters()
{
    Parameter first({1});
    Parameter second({1});

    first.gradients()[0] = 3.0;
    second.gradients()[0] = 4.0;

    GradientController controller(1.0);

    const auto report =
        controller.apply({&first, &second});

    near(report.norm_before, 5.0, 1e-12);
    near(first.gradients()[0], 0.6, 1e-12);
    near(second.gradients()[0], 0.8, 1e-12);

    require(report.parameter_blocks == 2,
            "Incorrect block count");
    require(report.parameter_count == 2,
            "Incorrect element count");
}

void test_measure_is_read_only()
{
    Parameter parameter({2});
    parameter.gradients()[0] = 3.0;
    parameter.gradients()[1] = 4.0;

    GradientController controller(1.0);
    const auto report = controller.measure({&parameter});

    near(report.norm_before, 5.0, 1e-12);
    near(parameter.gradients()[0], 3.0, 0.0);
    near(parameter.gradients()[1], 4.0, 0.0);
    require(!report.clipped, "Measurement changed state");
}

void test_threshold_and_zero()
{
    Parameter parameter({2});
    GradientController controller(1.0);

    auto report = controller.apply({&parameter});
    near(report.norm_before, 0.0, 0.0);
    near(report.applied_scale, 1.0, 0.0);
    require(!report.clipped, "Zero gradient clipped");

    parameter.gradients()[0] = 0.6;
    parameter.gradients()[1] = 0.8;

    report = controller.apply({&parameter});

    near(report.applied_scale, 1.0, 0.0);
    require(!report.clipped,
            "Boundary gradient unexpectedly clipped");
}

void test_invalid_registration()
{
    Parameter parameter({2});
    GradientController controller(1.0);

    rejects<std::invalid_argument>([&] {
        controller.measure({});
    });

    rejects<std::invalid_argument>([&] {
        controller.apply({&parameter, &parameter});
    });

    rejects<std::invalid_argument>([&] {
        controller.apply({&parameter, nullptr});
    });
}

void test_nan_is_atomic()
{
    Parameter first({1});
    Parameter second({1});

    first.gradients()[0] = 9.0;
    second.gradients()[0] =
        std::numeric_limits<double>::quiet_NaN();

    GradientController controller(1.0);

    rejects<std::invalid_argument>([&] {
        controller.apply({&first, &second});
    });

    near(first.gradients()[0], 9.0, 0.0);
    require(std::isnan(second.gradients()[0]),
            "Invalid gradient unexpectedly modified");
}

void test_large_finite_values()
{
    Parameter parameter({2});
    parameter.gradients()[0] = 1e200;
    parameter.gradients()[1] = 1e200;

    GradientController controller(1.0);

    // A naive sum of squares would overflow.
    const auto report = controller.apply({&parameter});

    require(std::isfinite(report.norm_before),
            "Norm overflowed");
    near(report.norm_after, 1.0, 1e-12);
    near(parameter.gradients()[0],
         std::sqrt(0.5), 1e-12);
    near(parameter.gradients()[1],
         std::sqrt(0.5), 1e-12);
}

void test_unrepresentable_norm_rejected()
{
    Parameter parameter({2});
    const double maximum =
        std::numeric_limits<double>::max();

    parameter.gradients()[0] = maximum;
    parameter.gradients()[1] = maximum;

    GradientController controller(1.0);

    rejects<std::overflow_error>([&] {
        controller.apply({&parameter});
    });

    require(parameter.gradients()[0] == maximum &&
            parameter.gradients()[1] == maximum,
            "Failed clipping modified gradients");
}

void test_invalid_threshold()
{
    rejects<std::invalid_argument>([] {
        GradientController controller(0.0);
    });

    rejects<std::invalid_argument>([] {
        GradientController controller(-1.0);
    });

    rejects<std::invalid_argument>([] {
        GradientController controller(
            std::numeric_limits<double>::infinity()
        );
    });
}

} // namespace

int main()
{
    try {
        test_known_answer();
        test_global_across_parameters();
        test_measure_is_read_only();
        test_threshold_and_zero();
        test_invalid_registration();
        test_nan_is_atomic();
        test_large_finite_values();
        test_unrepresentable_norm_rejected();
        test_invalid_threshold();

        std::cout
            << "[PASS] NN.1.07C Euclidean Gradient Controller\n";

        return 0;

    } catch (const std::exception& error) {
        std::cerr
            << "[FAIL] NN.1.07C: "
            << error.what() << '\n';

        return 1;
    }
}
