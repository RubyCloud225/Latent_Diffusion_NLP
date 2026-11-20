
#include "latent/nn/Conv2D.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using latent::core::Tensor;
using latent::nn::Conv2D;

namespace {

void require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(
    double actual,
    double expected,
    const std::string& message,
    double tolerance = 1e-7
)
{
    require(
        std::isfinite(actual) &&
        std::isfinite(expected) &&
        std::abs(actual - expected) <=
            tolerance * std::max(1.0, std::abs(expected)),
        message
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

/**
 * @brief Scalar loss used for independent numerical gradients.
 *
 * L = sum(output[i] * upstream[i]).
 */
double loss(
    Conv2D& conv,
    const Tensor& input,
    const Tensor& upstream
)
{
    const Tensor output = conv.forward(input);

    require(
        output.same_shape(upstream),
        "Loss output shape mismatch"
    );

    double result = 0.0;

    for (std::size_t i = 0; i < output.size(); ++i) {
        result += output[i] * upstream[i];
    }

    return result;
}

void test_known_answer()
{
    Conv2D conv(1, 1, 2, 2);
    conv.weights().values().fill(1.0);

    Tensor input({1, 1, 3, 3});

    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<double>(i + 1);
    }

    const Tensor output = conv.forward(input);

    Tensor upstream(output.shape());
    upstream.fill(1.0);

    const Tensor dx = conv.backward(upstream);

    // Each input contributes to as many output
    // windows as cover its spatial position.
    const double expected_dx[9] = {
        1, 2, 1,
        2, 4, 2,
        1, 2, 1
    };

    for (std::size_t i = 0; i < 9; ++i) {
        near(dx[i], expected_dx[i], "Input gradient");
    }

    const double expected_dw[4] = {
        12, 16,
        24, 28
    };

    for (std::size_t i = 0; i < 4; ++i) {
        near(
            conv.weights().gradients()[i],
            expected_dw[i],
            "Kernel gradient"
        );
    }

    near(
        conv.bias().gradients()[0],
        4.0,
        "Bias gradient"
    );
}

void test_gradient_accumulation()
{
    Conv2D conv(1, 1, 1, 1);
    conv.weights().values()[0] = 2.0;

    Tensor input({1, 1, 2, 2});
    input.fill(3.0);

    Tensor upstream({1, 1, 2, 2});
    upstream.fill(1.0);

    conv.forward(input);
    conv.backward(upstream);

    near(
        conv.weights().gradients()[0],
        12.0,
        "First weight accumulation"
    );

    near(
        conv.bias().gradients()[0],
        4.0,
        "First bias accumulation"
    );

    conv.backward(upstream);

    near(
        conv.weights().gradients()[0],
        24.0,
        "Second weight accumulation"
    );

    near(
        conv.bias().gradients()[0],
        8.0,
        "Second bias accumulation"
    );

    conv.zero_grad();

    near(
        conv.weights().gradients()[0],
        0.0,
        "Weight zero_grad failed"
    );

    near(
        conv.bias().gradients()[0],
        0.0,
        "Bias zero_grad failed"
    );
}

void test_batched_gradients()
{
    Conv2D conv(1, 2, 1, 1);

    conv.weights().values()[0] = 2.0;
    conv.weights().values()[1] = 3.0;

    Tensor input({2, 1, 1, 1});
    input[0] = 4.0;
    input[1] = 5.0;

    conv.forward(input);

    Tensor upstream({2, 2, 1, 1});
    upstream.fill(1.0);

    const Tensor dx = conv.backward(upstream);

    near(dx[0], 5.0, "Batch 0 input gradient");
    near(dx[1], 5.0, "Batch 1 input gradient");

    near(conv.weights().gradients()[0],
         9.0, "Output channel 0 weight gradient");
    near(conv.weights().gradients()[1],
         9.0, "Output channel 1 weight gradient");

    near(conv.bias().gradients()[0],
         2.0, "Output channel 0 bias gradient");
    near(conv.bias().gradients()[1],
         2.0, "Output channel 1 bias gradient");
}

void test_finite_differences()
{
    // Multi-channel and multi-output configuration
    // with stride and zero padding.
    Conv2D conv(2, 2, 2, 2, 2, 1);

    Tensor input({2, 2, 3, 3});

    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] =
            0.1 * static_cast<double>(i + 1);
    }

    for (std::size_t i = 0;
         i < conv.weights().values().size(); ++i) {
        conv.weights().values()[i] =
            0.02 * static_cast<double>(i + 1);
    }

    conv.bias().values()[0] = 0.1;
    conv.bias().values()[1] = -0.2;

    const Tensor output = conv.forward(input);
    Tensor upstream(output.shape());

    for (std::size_t i = 0; i < upstream.size(); ++i) {
        upstream[i] =
            0.1 + 0.01 * static_cast<double>(i);
    }

    conv.zero_grad();
    const Tensor analytical_dx = conv.backward(upstream);

    const Tensor analytical_dw =
        conv.weights().gradients();

    const Tensor analytical_db =
        conv.bias().gradients();

    const double h = 1e-6;

    // Input gradients.
    for (std::size_t i = 0; i < input.size(); ++i) {
        Tensor plus = input;
        Tensor minus = input;

        plus[i] += h;
        minus[i] -= h;

        const double numerical =
            (loss(conv, plus, upstream) -
             loss(conv, minus, upstream)) / (2.0 * h);

        near(
            analytical_dx[i],
            numerical,
            "Input finite difference"
        );
    }

    // Kernel gradients.
    for (std::size_t i = 0;
         i < conv.weights().values().size(); ++i) {

        const double original =
            conv.weights().values()[i];

        conv.weights().values()[i] = original + h;
        const double plus = loss(conv, input, upstream);

        conv.weights().values()[i] = original - h;
        const double minus = loss(conv, input, upstream);

        conv.weights().values()[i] = original;

        const double numerical =
            (plus - minus) / (2.0 * h);

        near(
            analytical_dw[i],
            numerical,
            "Kernel finite difference"
        );
    }

    // Bias gradients.
    for (std::size_t i = 0;
         i < conv.bias().values().size(); ++i) {

        const double original =
            conv.bias().values()[i];

        conv.bias().values()[i] = original + h;
        const double plus = loss(conv, input, upstream);

        conv.bias().values()[i] = original - h;
        const double minus = loss(conv, input, upstream);

        conv.bias().values()[i] = original;

        const double numerical =
            (plus - minus) / (2.0 * h);

        near(
            analytical_db[i],
            numerical,
            "Bias finite difference"
        );
    }

    // Re-establish original forward state after
    // perturbations used for numerical testing.
    conv.forward(input);
}

void test_validation()
{
    Conv2D conv(1, 1, 2, 2);

    Tensor upstream({1, 1, 2, 2});

    expect_throw<std::logic_error>([&] {
        (void)conv.backward(upstream);
    });

    Tensor input({1, 1, 3, 3});
    conv.forward(input);

    Tensor wrong({1, 1, 3, 3});

    expect_throw<std::invalid_argument>([&] {
        (void)conv.backward(wrong);
    });

    upstream[0] =
        std::numeric_limits<double>::infinity();

    expect_throw<std::invalid_argument>([&] {
        (void)conv.backward(upstream);
    });

    require(
        conv.weights().gradients().all_finite(),
        "Invalid backward corrupted weight gradients"
    );

    require(
        conv.bias().gradients().all_finite(),
        "Invalid backward corrupted bias gradients"
    );
}

} // namespace

int main()
{
    try {
        test_known_answer();
        test_gradient_accumulation();
        test_batched_gradients();
        test_finite_differences();
        test_validation();

        std::cout
            << "[PASS] NN.1.05 Conv2D Backward\n";

        return 0;

    } catch (const std::exception& error) {
        std::cerr
            << "[FAIL] NN.1.05: "
            << error.what() << '\n';

        return 1;
    }
}
