#include "latent/nn/Activation.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using latent::core::Tensor;
using latent::nn::Layer;
using latent::nn::ReLU;
using latent::nn::SiLU;

namespace {

void require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(
    double actual, double expected, double tolerance,
    const std::string& message
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

/**
 * @brief Independently approximate dL/dx using central differences.
 *
 * For the scalar objective L(x) = upstream * activation(x), compute
 * upstream * (f(x+h) - f(x-h)) / (2h). Forward calls change the layer
 * cache, so check_gradient() restores the original input before backward().
 */
double numerical_gradient(
    Layer& layer, double x, double upstream, double h
)
{
    Tensor plus({1});
    Tensor minus({1});
    plus[0] = x + h;
    minus[0] = x - h;

    const double y_plus = layer.forward(plus)[0];
    const double y_minus = layer.forward(minus)[0];
    return upstream * (y_plus - y_minus) / (2.0 * h);
}

void check_gradient(
    Layer& layer, double x, double upstream,
    double h, double tolerance
)
{
    const double numerical =
        numerical_gradient(layer, x, upstream, h);

    Tensor input({1});
    input[0] = x;
    layer.forward(input);

    Tensor gradient({1});
    gradient[0] = upstream;
    const Tensor analytical = layer.backward(gradient);

    near(
        analytical[0], numerical, tolerance,
        "Activation gradient mismatch at x=" + std::to_string(x)
    );
}

void test_relu_numerical_gradients()
{
    ReLU layer;
    const std::vector<double> inputs = {
        -10.0, -2.0, -0.5, -1e-3,
        1e-3, 0.5, 2.0, 10.0
    };

    for (const double x : inputs) {
        check_gradient(layer, x, 1.75, 1e-6, 1e-8);
    }
    // The derivative does not exist at zero: test the selected
    // zero-subgradient convention separately.
}

void test_relu_zero_subgradient()
{
    ReLU layer;
    Tensor input({1});
    input[0] = 0.0;
    layer.forward(input);

    Tensor gradient({1});
    gradient[0] = 3.0;
    const Tensor dx = layer.backward(gradient);
    require(dx[0] == 0.0, "ReLU zero subgradient must be zero");
}

void test_silu_numerical_gradients()
{
    SiLU layer;
    const std::vector<double> inputs = {
        -20.0, -5.0, -2.0, -1.0, -0.1,
        -1e-3, 0.0, 1e-3, 0.1, 1.0,
        2.0, 5.0, 20.0
    };

    for (const double x : inputs) {
        check_gradient(layer, x, -2.25, 1e-5, 1e-7);
    }
}

void test_silu_extreme_gradients()
{
    SiLU layer;
    Tensor input({6});
    input[0] = -1000.0;
    input[1] = -100.0;
    input[2] = -40.0;
    input[3] = 40.0;
    input[4] = 100.0;
    input[5] = 1000.0;

    const Tensor output = layer.forward(input);
    require(output.all_finite(), "SiLU extreme forward is non-finite");

    Tensor upstream(input.shape());
    upstream.fill(1.0);
    const Tensor dx = layer.backward(upstream);
    require(dx.all_finite(), "SiLU extreme backward is non-finite");

    near(dx[0], 0.0, 1e-12, "SiLU -1000");
    near(dx[1], 0.0, 1e-12, "SiLU -100");
    near(dx[4], 1.0, 1e-12, "SiLU +100");
    near(dx[5], 1.0, 1e-12, "SiLU +1000");
}

void test_rank4_gradients()
{
    ReLU relu;
    SiLU silu;
    Tensor input({2, 3, 4, 5});

    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<double>(i) / 20.0 - 3.0;
    }

    Tensor upstream(input.shape());
    upstream.fill(0.75);
    const Tensor relu_output = relu.forward(input);
    const Tensor silu_output = silu.forward(input);
    const Tensor relu_dx = relu.backward(upstream);
    const Tensor silu_dx = silu.backward(upstream);

    require(relu_output.same_shape(input), "ReLU output shape mismatch");
    require(silu_output.same_shape(input), "SiLU output shape mismatch");
    require(relu_dx.same_shape(input), "ReLU gradient shape mismatch");
    require(silu_dx.same_shape(input), "SiLU gradient shape mismatch");
    require(relu_dx.all_finite(), "ReLU gradient non-finite");
    require(silu_dx.all_finite(), "SiLU gradient non-finite");
}

void test_failed_forward_preserves_cache()
{
    ReLU relu;
    SiLU silu;

    Tensor valid({1});
    valid[0] = 2.0;
    Tensor invalid({1});
    invalid[0] = std::numeric_limits<double>::quiet_NaN();
    Tensor upstream({1});
    upstream[0] = 1.0;

    relu.forward(valid);
    silu.forward(valid);

    bool relu_rejected = false;
    bool silu_rejected = false;
    try {
        (void)relu.forward(invalid);
    } catch (const std::invalid_argument&) {
        relu_rejected = true;
    }
    try {
        (void)silu.forward(invalid);
    } catch (const std::invalid_argument&) {
        silu_rejected = true;
    }
    require(relu_rejected, "ReLU accepted NaN");
    require(silu_rejected, "SiLU accepted NaN");

    const Tensor relu_dx = relu.backward(upstream);
    const Tensor silu_dx = silu.backward(upstream);
    near(relu_dx[0], 1.0, 1e-12, "ReLU cache corrupted");

    const double sigmoid = 1.0 / (1.0 + std::exp(-2.0));
    const double expected =
        sigmoid + 2.0 * sigmoid * (1.0 - sigmoid);
    near(silu_dx[0], expected, 1e-12, "SiLU cache corrupted");
}

} // namespace

int main()
{
    try {
        test_relu_numerical_gradients();
        test_relu_zero_subgradient();
        test_silu_numerical_gradients();
        test_silu_extreme_gradients();
        test_rank4_gradients();
        test_failed_forward_preserves_cache();

        std::cout << "[PASS] NN.1.03C Activation numerical validation\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] NN.1.03C: "
                  << error.what() << '\n';
        return 1;
    }
}
