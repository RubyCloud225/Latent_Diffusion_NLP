
#include "latent/nn/Linear.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using latent::core::Tensor;
using latent::nn::Linear;

namespace {

void require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(double actual, double expected, double tol = 1e-10)
{
    require(
        std::isfinite(actual) &&
        std::abs(actual - expected) <= tol,
        "Numerical mismatch"
    );
}

template<typename Exception, typename Function>
void expect_throw(Function function)
{
    bool caught = false;

    try {
        function();
    } catch (const Exception&) {
        caught = true;
    }

    require(caught, "Expected exception not thrown");
}

void initialise(Linear& layer)
{
    // W = [[1, 2], [3, 4]]
    layer.weights().values()[0] = 1.0;
    layer.weights().values()[1] = 2.0;
    layer.weights().values()[2] = 3.0;
    layer.weights().values()[3] = 4.0;

    layer.bias().values()[0] = 0.5;
    layer.bias().values()[1] = -0.5;
}

void test_forward()
{
    Linear layer(2, 2);
    initialise(layer);

    Tensor x({2});
    x[0] = 2.0;
    x[1] = 3.0;

    const Tensor y = layer.forward(x);

    near(y[0], 8.5);
    near(y[1], 17.5);
}

void test_backward()
{
    Linear layer(2, 2);
    initialise(layer);

    Tensor x({2});
    x[0] = 2.0;
    x[1] = 3.0;

    layer.forward(x);

    Tensor grad({2});
    grad[0] = 1.0;
    grad[1] = 2.0;

    const Tensor dx = layer.backward(grad);

    near(dx[0], 7.0);
    near(dx[1], 10.0);

    const auto& dw = layer.weights().gradients();
    const auto& db = layer.bias().gradients();

    near(dw[0], 2.0);
    near(dw[1], 3.0);
    near(dw[2], 4.0);
    near(dw[3], 6.0);

    near(db[0], 1.0);
    near(db[1], 2.0);
}

void test_batch()
{
    Linear layer(2, 2);
    initialise(layer);

    Tensor x({2, 2});

    x[0] = 2.0;
    x[1] = 3.0;
    x[2] = 1.0;
    x[3] = 1.0;

    const Tensor y = layer.forward(x);

    require(
        y.shape() == Tensor::Shape({2, 2}),
        "Incorrect batch shape"
    );

    near(y[0], 8.5);
    near(y[1], 17.5);
    near(y[2], 3.5);
    near(y[3], 6.5);

    Tensor grad({2, 2});
    grad.fill(1.0);

    const Tensor dx = layer.backward(grad);

    require(dx.same_shape(x), "Input gradient shape changed");

    near(dx[0], 4.0);
    near(dx[1], 6.0);
    near(dx[2], 4.0);
    near(dx[3], 6.0);

    const auto& dw = layer.weights().gradients();
    const auto& db = layer.bias().gradients();

    near(dw[0], 3.0);
    near(dw[1], 4.0);
    near(dw[2], 3.0);
    near(dw[3], 4.0);
    near(db[0], 2.0);
    near(db[1], 2.0);
}

double objective(Linear& layer, const Tensor& x)
{
    // L = sum(y); hence dL/dy = 1.
    const Tensor y = layer.forward(x);
    double result = 0.0;

    for (std::size_t i = 0; i < y.size(); ++i) {
        result += y[i];
    }

    return result;
}

void test_numerical_gradients()
{
    Linear layer(2, 2);
    initialise(layer);

    Tensor x({2});
    x[0] = 0.75;
    x[1] = -1.25;

    layer.zero_grad();
    const Tensor y = layer.forward(x);

    Tensor grad(y.shape());
    grad.fill(1.0);

    const Tensor dx = layer.backward(grad);

    const double h = 1e-6;
    const double tol = 1e-8;

    // Independent finite-difference weight check.
    for (std::size_t i = 0; i < 4; ++i) {
        const double original = layer.weights().values()[i];

        layer.weights().values()[i] = original + h;
        const double plus = objective(layer, x);

        layer.weights().values()[i] = original - h;
        const double minus = objective(layer, x);

        layer.weights().values()[i] = original;

        const double numerical = (plus - minus) / (2.0 * h);

        near(
            layer.weights().gradients()[i],
            numerical,
            tol
        );
    }

    // Bias gradient check.
    for (std::size_t i = 0; i < 2; ++i) {
        const double original = layer.bias().values()[i];

        layer.bias().values()[i] = original + h;
        const double plus = objective(layer, x);

        layer.bias().values()[i] = original - h;
        const double minus = objective(layer, x);

        layer.bias().values()[i] = original;

        near(
            layer.bias().gradients()[i],
            (plus - minus) / (2.0 * h),
            tol
        );
    }

    // Input-gradient check.
    for (std::size_t i = 0; i < x.size(); ++i) {
        Tensor plus_x = x;
        Tensor minus_x = x;

        plus_x[i] += h;
        minus_x[i] -= h;

        const double plus = objective(layer, plus_x);
        const double minus = objective(layer, minus_x);

        near(dx[i], (plus - minus) / (2.0 * h), tol);
    }
}

void test_gradient_accumulation()
{
    Linear layer(2, 2);
    initialise(layer);

    Tensor x({2});
    x.fill(1.0);

    Tensor grad({2});
    grad.fill(1.0);

    layer.forward(x);
    layer.backward(grad);
    layer.backward(grad);

    for (std::size_t i = 0; i < 4; ++i) {
        near(layer.weights().gradients()[i], 2.0);
    }

    layer.zero_grad();

    for (std::size_t i = 0; i < 4; ++i) {
        near(layer.weights().gradients()[i], 0.0);
    }

    for (std::size_t i = 0; i < 2; ++i) {
        near(layer.bias().gradients()[i], 0.0);
    }
}

void test_validation()
{
    expect_throw<std::invalid_argument>([] {
        Linear invalid(0, 2);
    });

    Linear layer(2, 2);

    Tensor wrong({3});
    expect_throw<std::invalid_argument>([&] {
        (void)layer.forward(wrong);
    });

    Tensor grad({2});
    expect_throw<std::logic_error>([&] {
        (void)layer.backward(grad);
    });

    Tensor x({2});
    layer.forward(x);

    Tensor wrong_grad({3});
    expect_throw<std::invalid_argument>([&] {
        (void)layer.backward(wrong_grad);
    });

    x[0] = std::numeric_limits<double>::infinity();

    expect_throw<std::invalid_argument>([&] {
        (void)layer.forward(x);
    });
}

} // namespace

int main()
{
    try {
        test_forward();
        test_backward();
        test_batch();
        test_numerical_gradients();
        test_gradient_accumulation();
        test_validation();

        std::cout << "[PASS] NN.1.02 Linear Layer\n";
        return 0;

    } catch (const std::exception& error) {
        std::cerr
            << "[FAIL] NN.1.02: "
            << error.what() << '\n';

        return 1;
    }
}
