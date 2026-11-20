#include "latent/nn/Activation.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using latent::core::Tensor;
using latent::nn::SiLU;

namespace {

void require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(double actual, double expected, double tolerance = 1e-12)
{
    require(
        std::isfinite(actual) &&
        std::abs(actual - expected) <= tolerance,
        "SiLU numerical mismatch"
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

void test_forward_known_answers()
{
    SiLU layer;
    Tensor input({5});
    input[0] = -2.0;
    input[1] = -1.0;
    input[2] = 0.0;
    input[3] = 1.0;
    input[4] = 2.0;

    const Tensor original = input;
    const Tensor output = layer.forward(input);

    near(output[0], -0.2384058440442351);
    near(output[1], -0.2689414213699951);
    near(output[2], 0.0);
    near(output[3], 0.7310585786300049);
    near(output[4], 1.7615941559557649);

    require(output.same_shape(input), "SiLU changed input shape");
    for (std::size_t i = 0; i < input.size(); ++i) {
        near(input[i], original[i]);
    }
}

void test_backward_known_answers()
{
    SiLU layer;
    Tensor input({3});
    input[0] = -1.0;
    input[1] = 0.0;
    input[2] = 1.0;
    layer.forward(input);

    Tensor grad({3});
    grad.fill(2.0);
    const Tensor dx = layer.backward(grad);

    near(dx[0], 0.1446589762570265);
    near(dx[1], 1.0);
    near(dx[2], 1.8553410237429734);
}

void test_extreme_values()
{
    SiLU layer;
    Tensor input({4});
    input[0] = -1000.0;
    input[1] = -100.0;
    input[2] = 100.0;
    input[3] = 1000.0;

    const Tensor output = layer.forward(input);
    require(output.all_finite(), "Extreme input generated non-finite output");
    near(output[0], 0.0);
    near(output[1], 0.0);
    near(output[2], 100.0);
    near(output[3], 1000.0);

    Tensor grad({4});
    grad.fill(1.0);
    const Tensor dx = layer.backward(grad);
    require(dx.all_finite(), "Extreme input generated non-finite gradient");
    near(dx[0], 0.0);
    near(dx[1], 0.0);
    near(dx[2], 1.0);
    near(dx[3], 1.0);
}

void test_shape_preservation()
{
    SiLU layer;
    Tensor input({2, 3, 4, 5});
    input.fill(0.5);
    const Tensor output = layer.forward(input);
    require(output.same_shape(input), "SiLU changed rank-4 shape");

    Tensor grad(input.shape());
    grad.fill(1.0);
    const Tensor dx = layer.backward(grad);
    require(dx.same_shape(input), "SiLU backward changed shape");
    require(layer.parameters().empty(), "SiLU must not expose parameters");
}

void test_validation()
{
    SiLU layer;
    Tensor grad({2});
    expect_throw<std::logic_error>([&] {
        (void)layer.backward(grad);
    });

    Tensor input({2});
    layer.forward(input);

    Tensor wrong({3});
    expect_throw<std::invalid_argument>([&] {
        (void)layer.backward(wrong);
    });

    grad[0] = std::numeric_limits<double>::infinity();
    expect_throw<std::invalid_argument>([&] {
        (void)layer.backward(grad);
    });

    input[0] = std::numeric_limits<double>::quiet_NaN();
    expect_throw<std::invalid_argument>([&] {
        (void)layer.forward(input);
    });
}

} // namespace

int main()
{
    try {
        test_forward_known_answers();
        test_backward_known_answers();
        test_extreme_values();
        test_shape_preservation();
        test_validation();
        std::cout << "[PASS] NN.1.03B SiLU\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] NN.1.03B: " << error.what() << '\n';
        return 1;
    }
}
