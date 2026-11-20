#include "latent/nn/Activation.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>

using latent::core::Tensor;
using latent::nn::ReLU;

namespace {

void require(bool ok, const char* message)
{
    if (!ok) {
        throw std::runtime_error(message);
    }
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

void test_forward()
{
    ReLU layer;
    Tensor x({5});
    x[0] = -2.0;
    x[1] = -0.5;
    x[2] = 0.0;
    x[3] = 0.5;
    x[4] = 3.0;

    const Tensor y = layer.forward(x);
    require(y[0] == 0.0, "Negative input failed");
    require(y[1] == 0.0, "Negative input failed");
    require(y[2] == 0.0, "Zero input failed");
    require(y[3] == 0.5, "Positive input failed");
    require(y[4] == 3.0, "Positive input failed");
    require(x[0] == -2.0, "Input was modified");
}

void test_backward()
{
    ReLU layer;
    Tensor x({5});
    x[0] = -2.0;
    x[1] = -0.5;
    x[2] = 0.0;
    x[3] = 0.5;
    x[4] = 3.0;
    layer.forward(x);

    Tensor grad({5});
    grad.fill(2.0);
    const Tensor dx = layer.backward(grad);
    require(dx[0] == 0.0, "Negative gradient failed");
    require(dx[1] == 0.0, "Negative gradient failed");
    require(dx[2] == 0.0, "Zero derivative failed");
    require(dx[3] == 2.0, "Positive gradient failed");
    require(dx[4] == 2.0, "Positive gradient failed");
}

void test_shape_preservation()
{
    ReLU layer;
    Tensor x({2, 3, 4, 5});
    x.fill(0.25);
    const Tensor y = layer.forward(x);
    require(y.same_shape(x), "ReLU changed Tensor shape");
    require(y.all_finite(), "Non-finite output");

    Tensor grad(x.shape());
    grad.fill(1.0);
    const Tensor dx = layer.backward(grad);
    require(dx.same_shape(x), "Gradient shape changed");
    require(layer.parameters().empty(), "ReLU should have no parameters");
}

void test_validation()
{
    ReLU layer;
    Tensor grad({2});
    expect_throw<std::logic_error>([&] {
        (void)layer.backward(grad);
    });

    Tensor x({2});
    layer.forward(x);
    Tensor wrong({3});
    expect_throw<std::invalid_argument>([&] {
        (void)layer.backward(wrong);
    });

    grad[0] = std::numeric_limits<double>::infinity();
    expect_throw<std::invalid_argument>([&] {
        (void)layer.backward(grad);
    });

    x[0] = std::numeric_limits<double>::quiet_NaN();
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
        test_shape_preservation();
        test_validation();
        std::cout << "[PASS] NN.1.03A ReLU\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] NN.1.03A: "
                  << error.what() << '\n';
        return 1;
    }
}
