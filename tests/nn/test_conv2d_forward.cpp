
#include "latent/nn/Conv2D.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using latent::core::Tensor;
using latent::nn::Conv2D;

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(double actual, double expected)
{
    require(
        std::isfinite(actual) &&
        std::abs(actual - expected) < 1e-12,
        "Conv2D numerical mismatch"
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

void test_known_answer()
{
    Conv2D conv(1, 1, 2, 2);

    // 2x2 kernel of ones.
    conv.weights().values().fill(1.0);

    Tensor input({1, 1, 3, 3});

    for (std::size_t i = 0; i < 9; ++i) {
        input[i] = static_cast<double>(i + 1);
    }

    const Tensor output = conv.forward(input);

    require(
        output.shape() == Tensor::Shape({1, 1, 2, 2}),
        "Incorrect output shape"
    );

    near(output[0], 12.0);
    near(output[1], 16.0);
    near(output[2], 24.0);
    near(output[3], 28.0);
}

void test_multiple_channels()
{
    Conv2D conv(2, 1, 1, 1);

    conv.weights().values()[0] = 2.0;
    conv.weights().values()[1] = 3.0;

    conv.bias().values()[0] = 1.0;

    Tensor input({1, 2, 2, 2});

    // Channel 0.
    for (std::size_t i = 0; i < 4; ++i) {
        input[i] = 1.0;
    }

    // Channel 1.
    for (std::size_t i = 4; i < 8; ++i) {
        input[i] = 2.0;
    }

    const Tensor output = conv.forward(input);

    // 2*1 + 3*2 + 1 = 9.
    for (std::size_t i = 0; i < output.size(); ++i) {
        near(output[i], 9.0);
    }
}

void test_stride()
{
    Conv2D conv(1, 1, 2, 2, 2, 0);
    conv.weights().values().fill(1.0);

    Tensor input({1, 1, 4, 4});

    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<double>(i + 1);
    }

    const Tensor output = conv.forward(input);

    require(
        output.shape() == Tensor::Shape({1, 1, 2, 2}),
        "Incorrect strided output shape"
    );

    near(output[0], 14.0);
    near(output[1], 22.0);
    near(output[2], 46.0);
    near(output[3], 54.0);
}

void test_padding()
{
    Conv2D conv(1, 1, 3, 3, 1, 1);
    conv.weights().values().fill(1.0);

    Tensor input({1, 1, 2, 2});
    input.fill(1.0);

    const Tensor output = conv.forward(input);

    require(
        output.shape() == Tensor::Shape({1, 1, 2, 2}),
        "Incorrect padded output shape"
    );

    for (std::size_t i = 0; i < output.size(); ++i) {
        near(output[i], 4.0);
    }
}

void test_batch_and_output_channels()
{
    Conv2D conv(1, 2, 1, 1);

    conv.weights().values()[0] = 1.0;
    conv.weights().values()[1] = 2.0;

    Tensor input({2, 1, 2, 2});
    input.fill(3.0);

    const Tensor output = conv.forward(input);

    require(
        output.shape() == Tensor::Shape({2, 2, 2, 2}),
        "Incorrect batched output shape"
    );

    for (std::size_t n = 0; n < 2; ++n) {
        for (std::size_t oc = 0; oc < 2; ++oc) {
            for (std::size_t p = 0; p < 4; ++p) {
                const std::size_t idx =
                    (n * 2 + oc) * 4 + p;

                near(output[idx], oc == 0 ? 3.0 : 6.0);
            }
        }
    }
}

void test_input_preservation()
{
    Conv2D conv(1, 1, 1, 1);

    Tensor input({1, 1, 2, 2});
    input.fill(2.0);

    const Tensor original = input;

    (void)conv.forward(input);

    for (std::size_t i = 0; i < input.size(); ++i) {
        near(input[i], original[i]);
    }

    require(conv.parameters().size() == 2,
            "Expected weights and bias parameters");
}

void test_validation()
{
    expect_throw<std::invalid_argument>([] {
        Conv2D invalid(1, 1, 3, 3, 0, 0);
    });

    Conv2D conv(1, 1, 3, 3);

    Tensor wrong_rank({3, 3});

    expect_throw<std::invalid_argument>([&] {
        (void)conv.forward(wrong_rank);
    });

    Tensor wrong_channels({1, 2, 5, 5});

    expect_throw<std::invalid_argument>([&] {
        (void)conv.forward(wrong_channels);
    });

    Tensor too_small({1, 1, 2, 2});

    expect_throw<std::invalid_argument>([&] {
        (void)conv.forward(too_small);
    });

    Tensor nan_input({1, 1, 3, 3});

    nan_input[0] =
        std::numeric_limits<double>::quiet_NaN();

    expect_throw<std::invalid_argument>([&] {
        (void)conv.forward(nan_input);
    });

    Tensor valid({1, 1, 3, 3});
    conv.weights().values()[0] =
        std::numeric_limits<double>::infinity();

    expect_throw<std::invalid_argument>([&] {
        (void)conv.forward(valid);
    });

    Conv2D not_forwarded(1, 1, 3, 3);

    expect_throw<std::logic_error>([&] {
        (void)not_forwarded.backward(valid);
    });
}

} // namespace

int main()
{
    try {
        test_known_answer();
        test_multiple_channels();
        test_stride();
        test_padding();
        test_batch_and_output_channels();
        test_input_preservation();
        test_validation();

        std::cout
            << "[PASS] NN.1.04 Conv2D Forward\n";

        return 0;

    } catch (const std::exception& error) {
        std::cerr
            << "[FAIL] NN.1.04: "
            << error.what() << '\n';

        return 1;
    }
}
