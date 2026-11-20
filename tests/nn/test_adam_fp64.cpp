
#include "latent/nn/AdamFP64.hpp"
#include "latent/nn/Linear.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using latent::core::Tensor;
using latent::nn::AdamFP64;
using latent::nn::Linear;
using latent::nn::Parameter;

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(
    double actual,
    double expected,
    double tolerance = 1e-12
)
{
    require(
        std::isfinite(actual) &&
        std::abs(actual - expected) <= tolerance,
        "AdamFP64 numerical mismatch"
    );
}

template<typename Exception, typename Function>
void expect_throw(Function fn)
{
    bool caught = false;

    try {
        fn();
    } catch (const Exception&) {
        caught = true;
    }

    require(caught, "Expected exception not thrown");
}

void test_first_step()
{
    Parameter parameter({2});

    parameter.values()[0] = 1.0;
    parameter.values()[1] = -1.0;

    parameter.gradients()[0] = 2.0;
    parameter.gradients()[1] = -3.0;

    AdamFP64 adam({&parameter}, 0.01);
    adam.step();

    near(
        parameter.values()[0],
        1.0 - 0.01 * 2.0 / (2.0 + 1e-8)
    );

    near(
        parameter.values()[1],
        -1.0 + 0.01 * 3.0 / (3.0 + 1e-8)
    );

    require(adam.step_count() == 1,
            "Incorrect Adam step count");

    near(parameter.gradients()[0], 2.0);
    near(parameter.gradients()[1], -3.0);
}

void test_multiple_steps()
{
    Parameter parameter({1});
    parameter.values()[0] = 5.0;

    // beta1=beta2=0 gives an exact simple reference:
    // update = lr * gradient / (abs(gradient)+eps).
    AdamFP64 adam({&parameter}, 0.1, 0.0, 0.0, 1e-8);

    parameter.gradients()[0] = 2.0;
    adam.step();

    const double first =
        5.0 - 0.1 * 2.0 / (2.0 + 1e-8);

    near(parameter.values()[0], first);

    adam.zero_grad();
    parameter.gradients()[0] = -4.0;
    adam.step();

    const double second =
        first + 0.1 * 4.0 / (4.0 + 1e-8);

    near(parameter.values()[0], second);

    require(adam.step_count() == 2,
            "Second step not recorded");
}

void test_deterministic_replay()
{
    Parameter a({2});
    Parameter b({2});

    a.values()[0] = b.values()[0] = 1.0;
    a.values()[1] = b.values()[1] = -2.0;

    AdamFP64 first({&a});
    AdamFP64 second({&b});

    for (int step = 0; step < 5; ++step) {
        first.zero_grad();
        second.zero_grad();

        a.gradients()[0] = b.gradients()[0] =
            0.5 + step * 0.1;

        a.gradients()[1] = b.gradients()[1] =
            -0.3 - step * 0.2;

        first.step();
        second.step();
    }

    for (std::size_t i = 0; i < 2; ++i) {
        near(a.values()[i], b.values()[i], 0.0);
    }

    require(
        first.step_count() == second.step_count(),
        "Deterministic step count mismatch"
    );
}

void test_multiple_parameters()
{
    Parameter weights({2, 2});
    Parameter bias({2});

    weights.values().fill(1.0);
    bias.values().fill(2.0);

    weights.gradients().fill(1.0);
    bias.gradients().fill(-1.0);

    AdamFP64 adam({&weights, &bias}, 0.01);
    adam.step();

    for (std::size_t i = 0;
         i < weights.values().size(); ++i) {
        require(weights.values()[i] < 1.0,
                "Weight failed to decrease");
    }

    for (std::size_t i = 0;
         i < bias.values().size(); ++i) {
        require(bias.values()[i] > 2.0,
                "Bias failed to increase");
    }
}

void test_failed_step_is_atomic()
{
    Parameter good({1});
    Parameter bad({1});

    good.values()[0] = 10.0;
    bad.values()[0] = 20.0;

    good.gradients()[0] = 1.0;

    // Finite, but its square overflows FP64.
    bad.gradients()[0] = 1e308;

    AdamFP64 adam({&good, &bad});

    expect_throw<std::overflow_error>([&] {
        adam.step();
    });

    near(good.values()[0], 10.0);
    near(bad.values()[0], 20.0);

    require(adam.step_count() == 0,
            "Failed step advanced counter");

    // A subsequent valid step must succeed.
    bad.gradients()[0] = 1.0;
    adam.step();

    require(adam.step_count() == 1,
            "Recovery step failed");
}

void test_invalid_inputs()
{
    Parameter parameter({1});

    expect_throw<std::invalid_argument>([&] {
        AdamFP64 invalid({});
    });

    expect_throw<std::invalid_argument>([&] {
        AdamFP64 invalid({nullptr});
    });

    expect_throw<std::invalid_argument>([&] {
        AdamFP64 invalid({&parameter, &parameter});
    });

    expect_throw<std::invalid_argument>([&] {
        AdamFP64 invalid({&parameter}, 0.0);
    });

    expect_throw<std::invalid_argument>([&] {
        AdamFP64 invalid({&parameter}, 0.01, 1.0);
    });

    expect_throw<std::invalid_argument>([&] {
        AdamFP64 invalid(
            {&parameter},
            std::numeric_limits<double>::infinity()
        );
    });

    AdamFP64 adam({&parameter});

    parameter.gradients()[0] =
        std::numeric_limits<double>::quiet_NaN();

    expect_throw<std::invalid_argument>([&] {
        adam.step();
    });

    require(adam.step_count() == 0,
            "Invalid step changed state");
}

void test_reset()
{
    Parameter parameter({1});

    parameter.values()[0] = 1.0;
    parameter.gradients()[0] = 0.5;

    AdamFP64 adam({&parameter});
    adam.step();

    const double current_value = parameter.values()[0];

    adam.reset();

    require(adam.step_count() == 0,
            "Adam reset failed");

    near(parameter.values()[0], current_value);
    near(parameter.gradients()[0], 0.5);

    adam.step();

    near(
        parameter.values()[0],
        current_value -
            0.001 * 0.5 / (0.5 + 1e-8)
    );
}

void test_linear_integration()
{
    Linear layer(2, 1);

    layer.weights().values().fill(1.0);
    layer.bias().values().fill(0.0);

    AdamFP64 adam(layer.parameters(), 0.01);

    Tensor input({2});
    input[0] = 1.0;
    input[1] = 2.0;

    adam.zero_grad();

    const Tensor before = layer.forward(input);

    Tensor upstream({1});
    upstream[0] = 1.0;

    layer.backward(upstream);
    adam.step();

    const Tensor after = layer.forward(input);

    require(after[0] < before[0],
            "Adam did not update Linear output");
}

} // namespace

int main()
{
    try {
        test_first_step();
        test_multiple_steps();
        test_deterministic_replay();
        test_multiple_parameters();
        test_failed_step_is_atomic();
        test_invalid_inputs();
        test_reset();
        test_linear_integration();

        std::cout << "[PASS] NN.1.06 AdamFP64\n";
        return 0;

    } catch (const std::exception& error) {
        std::cerr
            << "[FAIL] NN.1.06: "
            << error.what() << '\n';

        return 1;
    }
}
