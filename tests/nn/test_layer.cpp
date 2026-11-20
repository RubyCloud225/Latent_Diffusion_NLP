
#include "latent/nn/Layer.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using latent::core::Tensor;
using latent::nn::Layer;
using latent::nn::Parameter;

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

/**
 * Test-only layer implementing:
 *
 * y = w * x
 *
 * dL/dx = w * dL/dy
 * dL/dw = sum(x * dL/dy)
 */
class ScaleLayer final : public Layer {
public:
    ScaleLayer() : weight_({1}) {
        weight_.values()[0] = 2.0;
    }

    Tensor forward(const Tensor& input) override {
        cached_input_ = input;
        has_forward_ = true;

        Tensor output(input.shape());

        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = weight_.values()[0] * input[i];
        }

        return output;
    }

    Tensor backward(const Tensor& grad_output) override {
        if (!has_forward_) {
            throw std::logic_error(
                "Backward requires forward"
            );
        }

        if (!grad_output.same_shape(cached_input_)) {
            throw std::invalid_argument(
                "Output gradient shape mismatch"
            );
        }

        Tensor grad_input(cached_input_.shape());
        Tensor grad_weight({1});
        grad_weight[0] = 0.0;

        for (std::size_t i = 0; i < grad_output.size(); ++i) {
            grad_input[i] =
                weight_.values()[0] * grad_output[i];

            grad_weight[0] +=
                cached_input_[i] * grad_output[i];
        }

        weight_.accumulate_gradient(grad_weight);
        return grad_input;
    }

    std::vector<Parameter*> parameters() override {
        return {&weight_};
    }

private:
    Parameter weight_;
    Tensor cached_input_{{1}};
    bool has_forward_ = false;
};

void test_forward_backward() {
    ScaleLayer layer;

    Tensor input({2});
    input[0] = 3.0;
    input[1] = 4.0;

    const Tensor output = layer.forward(input);

    require(output[0] == 6.0, "Incorrect forward result");
    require(output[1] == 8.0, "Incorrect forward result");

    Tensor grad_output({2});
    grad_output.fill(1.0);

    const Tensor grad_input = layer.backward(grad_output);

    require(grad_input[0] == 2.0, "Incorrect input gradient");
    require(grad_input[1] == 2.0, "Incorrect input gradient");

    require(
        layer.parameters()[0]->gradients()[0] == 7.0,
        "Incorrect parameter gradient"
    );

    // Gradients must accumulate across backward calls.
    layer.backward(grad_output);

    require(
        layer.parameters()[0]->gradients()[0] == 14.0,
        "Gradient accumulation failed"
    );

    layer.zero_grad();

    require(
        layer.parameters()[0]->gradients()[0] == 0.0,
        "zero_grad failed"
    );
}

void test_backward_before_forward() {
    ScaleLayer layer;
    Tensor grad({2});
    grad.fill(1.0);

    bool caught = false;

    try {
        (void)layer.backward(grad);
    } catch (const std::logic_error&) {
        caught = true;
    }

    require(caught, "Missing forward validation");
}

void test_gradient_shape_validation() {
    ScaleLayer layer;

    Tensor input({2});
    layer.forward(input);

    Tensor wrong({3});

    bool caught = false;

    try {
        (void)layer.backward(wrong);
    } catch (const std::invalid_argument&) {
        caught = true;
    }

    require(caught, "Missing gradient shape validation");
}

void test_nonfinite_gradient_rejection() {
    Parameter parameter({1});

    Tensor bad({1});
    bad[0] = std::numeric_limits<double>::infinity();

    bool caught = false;
    try {
        parameter.accumulate_gradient(bad);
    } catch (const std::invalid_argument&) {
        caught = true;
    }

    require(caught, "Non-finite gradient was accepted");
}

} // namespace

int main() {
    try {
        test_forward_backward();
        test_backward_before_forward();
        test_gradient_shape_validation();
        test_nonfinite_gradient_rejection();

        std::cout << "[PASS] NN.1.01 Layer interface\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] NN.1.01: "
                  << error.what() << '\n';
        return 1;
    }
}
