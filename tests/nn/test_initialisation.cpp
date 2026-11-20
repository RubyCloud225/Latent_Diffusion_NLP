
#include "latent/nn/training/Initialisation.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using latent::nn::Conv2D;
using latent::nn::Linear;
using latent::nn::training::Initialisation;
using latent::nn::training::InitialisationRecord;
using latent::nn::training::ParameterInitialiser;

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
    double tolerance,
    const std::string& message
)
{
    require(
        std::isfinite(actual) &&
        std::isfinite(expected) &&
        std::abs(actual - expected) <= tolerance,
        message
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

    require(caught, "Expected exception was not thrown");
}

bool identical(
    const latent::core::Tensor& a,
    const latent::core::Tensor& b
)
{
    if (!a.same_shape(b)) {
        return false;
    }

    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }

    return true;
}

bool differs(
    const latent::core::Tensor& a,
    const latent::core::Tensor& b
)
{
    require(a.same_shape(b), "Tensor shapes differ");

    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            return true;
        }
    }

    return false;
}

double sample_variance(const latent::core::Tensor& values)
{
    const std::size_t count = values.size();

    require(count > 1, "Variance requires multiple values");

    double mean = 0.0;

    for (std::size_t i = 0; i < count; ++i) {
        mean += values[i];
    }

    mean /= static_cast<double>(count);

    double sum = 0.0;

    for (std::size_t i = 0; i < count; ++i) {
        const double deviation = values[i] - mean;
        sum += deviation * deviation;
    }

    return sum / static_cast<double>(count - 1);
}

void test_identical_seed()
{
    Linear a(32, 16);
    Linear b(32, 16);

    ParameterInitialiser initialiser(42);

    const auto record_a = initialiser.initialise(
        a, "encoder.linear.0", Initialisation::HeNormal
    );

    const auto record_b = initialiser.initialise(
        b, "encoder.linear.0", Initialisation::HeNormal
    );

    require(
        identical(a.weights().values(), b.weights().values()),
        "Same seed produced different Linear weights"
    );

    require(
        record_a.derived_seed == record_b.derived_seed,
        "Derived seeds differ"
    );

    require(
        record_a.master_seed == 42,
        "Incorrect master seed"
    );

    require(
        record_a.layer_id == "encoder.linear.0",
        "Incorrect layer identifier"
    );

    require(
        record_a.algorithm_id == "initialisation_v1",
        "Unexpected initialisation algorithm version"
    );
}

void test_different_master_seeds()
{
    Linear a(32, 16);
    Linear b(32, 16);

    ParameterInitialiser first(42);
    ParameterInitialiser second(43);

    first.initialise(
        a, "encoder.linear.0", Initialisation::HeNormal
    );

    second.initialise(
        b, "encoder.linear.0", Initialisation::HeNormal
    );

    require(
        differs(a.weights().values(), b.weights().values()),
        "Different master seeds produced identical weights"
    );
}

void test_different_layer_ids()
{
    Linear a(32, 16);
    Linear b(32, 16);

    ParameterInitialiser initialiser(42);

    initialiser.initialise(
        a, "encoder.linear.0", Initialisation::XavierNormal
    );

    initialiser.initialise(
        b, "encoder.linear.1", Initialisation::XavierNormal
    );

    require(
        differs(a.weights().values(), b.weights().values()),
        "Different layer IDs produced identical weights"
    );
}

void test_initialisation_order()
{
    Linear first_a(16, 8);
    Linear second_a(16, 8);

    Linear first_b(16, 8);
    Linear second_b(16, 8);

    ParameterInitialiser initialiser(123456);

    initialiser.initialise(
        first_a, "layer.first", Initialisation::HeNormal
    );

    initialiser.initialise(
        second_a, "layer.second", Initialisation::HeNormal
    );

    // Reverse layer initialisation order.
    initialiser.initialise(
        second_b, "layer.second", Initialisation::HeNormal
    );

    initialiser.initialise(
        first_b, "layer.first", Initialisation::HeNormal
    );

    require(
        identical(
            first_a.weights().values(),
            first_b.weights().values()
        ),
        "First layer depends on initialisation order"
    );

    require(
        identical(
            second_a.weights().values(),
            second_b.weights().values()
        ),
        "Second layer depends on initialisation order"
    );
}

void test_he_normal_statistics()
{
    Linear layer(256, 256);

    ParameterInitialiser initialiser(42);

    const auto record = initialiser.initialise(
        layer, "statistics.he", Initialisation::HeNormal
    );

    const double expected_variance = 2.0 / 256.0;
    const double measured_variance =
        sample_variance(layer.weights().values());

    const double relative_error =
        std::abs(measured_variance - expected_variance) /
        expected_variance;

    require(
        relative_error < 0.05,
        "He initialisation variance outside 5% tolerance"
    );

    near(
        record.standard_deviation,
        std::sqrt(expected_variance),
        1e-15,
        "Incorrect He standard deviation"
    );

    require(
        record.fan_in == 256 &&
        record.fan_out == 256,
        "Incorrect Linear fan calculations"
    );
}

void test_xavier_normal_statistics()
{
    Linear layer(128, 256);

    ParameterInitialiser initialiser(99);

    const auto record = initialiser.initialise(
        layer, "statistics.xavier",
        Initialisation::XavierNormal
    );

    const double expected_variance =
        2.0 / (128.0 + 256.0);

    const double measured_variance =
        sample_variance(layer.weights().values());

    const double relative_error =
        std::abs(measured_variance - expected_variance) /
        expected_variance;

    require(
        relative_error < 0.05,
        "Xavier initialisation variance outside 5% tolerance"
    );

    near(
        record.standard_deviation,
        std::sqrt(expected_variance),
        1e-15,
        "Incorrect Xavier standard deviation"
    );
}

void test_conv2d_initialisation()
{
    Conv2D conv(3, 8, 3, 3);

    ParameterInitialiser initialiser(42);

    const InitialisationRecord record =
        initialiser.initialise(
            conv, "encoder.conv.0", Initialisation::HeNormal
        );

    // Fan-in  = 3 * 3 * 3 = 27
    // Fan-out = 8 * 3 * 3 = 72
    require(
        record.fan_in == 27,
        "Incorrect convolution fan-in"
    );

    require(
        record.fan_out == 72,
        "Incorrect convolution fan-out"
    );

    near(
        record.standard_deviation,
        std::sqrt(2.0 / 27.0),
        1e-15,
        "Incorrect Conv2D He scaling"
    );

    require(
        conv.weights().values().all_finite(),
        "Conv2D weights contain non-finite values"
    );

    Conv2D duplicate(3, 8, 3, 3);

    initialiser.initialise(
        duplicate, "encoder.conv.0", Initialisation::HeNormal
    );

    require(
        identical(
            conv.weights().values(),
            duplicate.weights().values()
        ),
        "Conv2D initialisation is not reproducible"
    );
}

void test_zero_biases()
{
    Linear linear(16, 8);
    Conv2D conv(3, 8, 3, 3);

    linear.bias().values().fill(5.0);
    conv.bias().values().fill(-3.0);

    ParameterInitialiser initialiser(42);

    initialiser.initialise(
        linear, "linear.bias", Initialisation::HeNormal
    );

    initialiser.initialise(
        conv, "conv.bias", Initialisation::XavierNormal
    );

    for (std::size_t i = 0;
         i < linear.bias().values().size(); ++i) {
        near(
            linear.bias().values()[i], 0.0, 0.0,
            "Linear bias not zero"
        );
    }

    for (std::size_t i = 0;
         i < conv.bias().values().size(); ++i) {
        near(
            conv.bias().values()[i], 0.0, 0.0,
            "Conv2D bias not zero"
        );
    }
}

void test_gradient_reset()
{
    Linear layer(8, 4);

    layer.weights().gradients().fill(2.0);
    layer.bias().gradients().fill(-1.0);

    ParameterInitialiser initialiser(42);

    initialiser.initialise(
        layer, "reset.test", Initialisation::HeNormal
    );

    for (std::size_t i = 0;
         i < layer.weights().gradients().size(); ++i) {
        near(
            layer.weights().gradients()[i], 0.0, 0.0,
            "Weight gradient not reset"
        );
    }

    for (std::size_t i = 0;
         i < layer.bias().gradients().size(); ++i) {
        near(
            layer.bias().gradients()[i], 0.0, 0.0,
            "Bias gradient not reset"
        );
    }
}

void test_invalid_method_is_atomic()
{
    Linear layer(8, 4);

    layer.weights().values().fill(2.0);
    layer.bias().values().fill(3.0);

    layer.weights().gradients().fill(4.0);
    layer.bias().gradients().fill(5.0);

    const auto original_weights = layer.weights().values();
    const auto original_bias = layer.bias().values();
    const auto original_weight_gradients =
        layer.weights().gradients();
    const auto original_bias_gradients =
        layer.bias().gradients();

    ParameterInitialiser initialiser(42);

    expect_throw<std::invalid_argument>([&] {
        initialiser.initialise(
            layer,
            "invalid.method",
            static_cast<Initialisation>(999)
        );
    });

    require(
        identical(layer.weights().values(), original_weights),
        "Invalid method modified weights"
    );

    require(
        identical(layer.bias().values(), original_bias),
        "Invalid method modified biases"
    );

    require(
        identical(
            layer.weights().gradients(),
            original_weight_gradients
        ),
        "Invalid method modified weight gradients"
    );

    require(
        identical(
            layer.bias().gradients(),
            original_bias_gradients
        ),
        "Invalid method modified bias gradients"
    );
}

void test_invalid_layer_id()
{
    Linear layer(8, 4);

    layer.weights().values().fill(7.0);

    const auto original = layer.weights().values();

    ParameterInitialiser initialiser(42);

    expect_throw<std::invalid_argument>([&] {
        initialiser.initialise(
            layer, "", Initialisation::HeNormal
        );
    });

    require(
        identical(layer.weights().values(), original),
        "Invalid layer identifier modified weights"
    );
}

void test_finite_nonzero_weights()
{
    Linear layer(64, 32);

    ParameterInitialiser initialiser(42);

    initialiser.initialise(
        layer, "nonzero.test", Initialisation::HeNormal
    );

    require(
        layer.weights().values().all_finite(),
        "Initialised weights contain NaN or Inf"
    );

    bool nonzero = false;

    for (std::size_t i = 0;
         i < layer.weights().values().size(); ++i) {
        if (layer.weights().values()[i] != 0.0) {
            nonzero = true;
            break;
        }
    }

    require(nonzero, "All initialised weights are zero");
}

} // namespace

int main()
{
    try {
        test_identical_seed();
        test_different_master_seeds();
        test_different_layer_ids();
        test_initialisation_order();
        test_he_normal_statistics();
        test_xavier_normal_statistics();
        test_conv2d_initialisation();
        test_zero_biases();
        test_gradient_reset();
        test_invalid_method_is_atomic();
        test_invalid_layer_id();
        test_finite_nonzero_weights();

        std::cout
            << "[PASS] NN.1.07B Deterministic Initialisation\n";

        return 0;

    } catch (const std::exception& error) {
        std::cerr
            << "[FAIL] NN.1.07B: "
            << error.what() << '\n';

        return 1;
    }
}
