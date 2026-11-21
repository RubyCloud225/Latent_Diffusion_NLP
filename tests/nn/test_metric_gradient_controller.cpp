
#include "latent/nn/training/MetricGradientController.hpp"

#include <cmath>
#include <iostream>
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

} // namespace

int main()
{
    try {
        test_known_answer();
        test_zero_gradient();
        test_block_normalisation();
        test_invalid_registration();

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
