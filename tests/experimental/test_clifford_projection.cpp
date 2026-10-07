#include "../../latent/core/FP16.hpp"
#include "../../latent/experimental/CliffordProjection.hpp"
#include "../../latent/nn/Adam.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void expect_near(
    float actual,
    float expected,
    float tolerance
)
{
    const float difference =
        std::fabs(actual - expected);

    if (difference > tolerance) {
        std::cerr
            << "EXPECT_NEAR failed: "
            << "actual=" << actual
            << " expected=" << expected
            << " difference=" << difference
            << " tolerance=" << tolerance
            << '\n';
    }

    assert(difference <= tolerance);
}


latent::experimental::CliffordProjection::Matrix
make_identity_projection()
{
    using Projection =
        latent::experimental::CliffordProjection;

    Projection::Matrix matrix{};

    for (
        std::size_t blade = 0;
        blade < Projection::BladeCount;
        ++blade
    ) {
        matrix[
            blade * Projection::SourceDimensions +
            blade
        ] = 1.0f;
    }

    return matrix;
}


std::vector<float>
make_test_input()
{
    using Projection =
        latent::experimental::CliffordProjection;

    std::vector<float> input(
        Projection::SourceDimensions
    );

    for (
        std::size_t i = 0;
        i < input.size();
        ++i
    ) {
        input[i] =
            static_cast<float>(i + 1);
    }

    return input;
}


void test_known_projection()
{
    using Projection =
        latent::experimental::CliffordProjection;

    std::cout
        << "\n=== Known Projection Test ===\n";

    Projection projection(
        make_identity_projection()
    );

    const auto input =
        make_test_input();

    const auto result =
        projection.forward(input);

    for (
        std::size_t i = 0;
        i < Projection::BladeCount;
        ++i
    ) {
        const float expected =
            static_cast<float>(i + 1);

        std::cout
            << "blade[" << i << "] "
            << "actual=" << result.fp32[i]
            << " expected=" << expected
            << '\n';

        expect_near(
            result.fp32[i],
            expected,
            1e-6f
        );
    }
}


void test_fp16_coefficients()
{
    using Projection =
        latent::experimental::CliffordProjection;

    std::cout
        << "\n=== FP16 Coefficient Test ===\n";

    Projection projection(
        make_identity_projection()
    );

    const auto result =
        projection.forward(
            make_test_input()
        );

    for (
        std::size_t i = 0;
        i < Projection::BladeCount;
        ++i
    ) {
        const float restored =
            latent::core::fp16_to_float(
                result.fp16[i]
            );

        const float expected =
            static_cast<float>(i + 1);

        const float difference =
            std::fabs(
                restored -
                expected
            );

        std::cout
            << "blade[" << i << "] "
            << "fp32=" << result.fp32[i]
            << " fp16-restored=" << restored
            << " diff=" << difference
            << '\n';

        expect_near(
            restored,
            expected,
            1e-3f
        );
    }
}


void test_reconstruction()
{
    using Projection =
        latent::experimental::CliffordProjection;

    std::cout
        << "\n=== Reconstruction Test ===\n";

    Projection projection(
        make_identity_projection()
    );

    const auto input =
        make_test_input();

    const auto compressed =
        projection.forward(input);

    const auto reconstructed =
        projection.reconstruct(
            compressed
        );

    assert(
        reconstructed.size() ==
        Projection::SourceDimensions
    );

    for (
        std::size_t i = 0;
        i < Projection::BladeCount;
        ++i
    ) {
        const float expected =
            static_cast<float>(i + 1);

        std::cout
            << "reconstructed[" << i << "] "
            << "actual=" << reconstructed[i]
            << " expected=" << expected
            << '\n';

        expect_near(
            reconstructed[i],
            expected,
            1e-6f
        );
    }

    for (
        std::size_t i = Projection::BladeCount;
        i < Projection::SourceDimensions;
        ++i
    ) {
        expect_near(
            reconstructed[i],
            0.0f,
            1e-6f
        );
    }

    std::cout
        << "Remaining dimensions correctly reconstructed as 0.\n";
}


void test_invalid_input_size()
{
    using Projection =
        latent::experimental::CliffordProjection;

    std::cout
        << "\n=== Invalid Input Test ===\n";

    Projection projection(
        make_identity_projection()
    );

    bool threw = false;

    try {
        projection.forward(
            std::vector<float>(32, 1.0f)
        );
    }
    catch (const std::invalid_argument& error) {
        threw = true;

        std::cout
            << "Caught expected exception: "
            << error.what()
            << '\n';
    }

    assert(threw);
}


void test_backward_gradients()
{
    using Projection =
        latent::experimental::CliffordProjection;

    std::cout
        << "\n=== Numerical Gradient Test ===\n";

    auto matrix =
        make_identity_projection();

    Projection projection(matrix);

    std::vector<float> input(
        Projection::SourceDimensions
    );

    for (
        std::size_t i = 0;
        i < input.size();
        ++i
    ) {
        input[i] =
            0.01f *
            static_cast<float>(i + 1);
    }

    std::array<
        float,
        Projection::BladeCount
    > grad_output{
        1.0f,
        -0.5f,
        0.25f,
        -0.75f,
        0.5f,
        1.25f,
        -1.0f,
        0.125f
    };

    const auto analytic =
        projection.backward(
            input,
            grad_output
        );

    constexpr float epsilon =
        1e-3f;

    constexpr float tolerance =
        1e-3f;

    auto compute_loss =
        [&](Projection& model) -> float
    {
        const auto output =
            model.forward(input);

        float loss = 0.0f;

        for (
            std::size_t i = 0;
            i < Projection::BladeCount;
            ++i
        ) {
            loss +=
                output.fp32[i] *
                grad_output[i];
        }

        return loss;
    };

    std::cout
        << "\nWeight gradients\n";

    std::cout
        << std::left
        << std::setw(14) << "Parameter"
        << std::setw(18) << "Analytical"
        << std::setw(18) << "Numerical"
        << std::setw(18) << "Difference"
        << '\n';

    std::cout
        << std::string(68, '-')
        << '\n';

    const std::size_t test_weight_indices[] = {
        0,
        1,
        63,
        64,
        127,
        255,
        511
    };

    for (
        const std::size_t index :
        test_weight_indices
    ) {
        auto plus_matrix =
            projection.weights();

        plus_matrix[index] += epsilon;

        Projection plus_model(
            plus_matrix
        );

        auto minus_matrix =
            projection.weights();

        minus_matrix[index] -= epsilon;

        Projection minus_model(
            minus_matrix
        );

        const float loss_plus =
            compute_loss(
                plus_model
            );

        const float loss_minus =
            compute_loss(
                minus_model
            );

        const float numerical =
            (
                loss_plus -
                loss_minus
            ) /
            (2.0f * epsilon);

        const float analytical =
            analytic.grad_weights[
                index
            ];

        const float difference =
            std::fabs(
                analytical -
                numerical
            );

        std::cout
            << std::left
            << std::setw(14)
            << (
                "weight[" +
                std::to_string(index) +
                "]"
            )
            << std::setw(18)
            << analytical
            << std::setw(18)
            << numerical
            << std::setw(18)
            << difference
            << '\n';

        expect_near(
            analytical,
            numerical,
            tolerance
        );
    }

    std::cout
        << "\nInput gradients\n";

    std::cout
        << std::left
        << std::setw(14) << "Parameter"
        << std::setw(18) << "Analytical"
        << std::setw(18) << "Numerical"
        << std::setw(18) << "Difference"
        << '\n';

    std::cout
        << std::string(68, '-')
        << '\n';

    const std::size_t test_input_indices[] = {
        0,
        1,
        7,
        8,
        31,
        63
    };

    for (
        const std::size_t dimension :
        test_input_indices
    ) {
        auto input_plus =
            input;

        input_plus[dimension] +=
            epsilon;

        auto input_minus =
            input;

        input_minus[dimension] -=
            epsilon;

        const auto output_plus =
            projection.forward(
                input_plus
            );

        const auto output_minus =
            projection.forward(
                input_minus
            );

        float loss_plus = 0.0f;
        float loss_minus = 0.0f;

        for (
            std::size_t i = 0;
            i < Projection::BladeCount;
            ++i
        ) {
            loss_plus +=
                output_plus.fp32[i] *
                grad_output[i];

            loss_minus +=
                output_minus.fp32[i] *
                grad_output[i];
        }

        const float numerical =
            (
                loss_plus -
                loss_minus
            ) /
            (2.0f * epsilon);

        const float analytical =
            analytic.grad_input[
                dimension
            ];

        const float difference =
            std::fabs(
                analytical -
                numerical
            );

        std::cout
            << std::left
            << std::setw(14)
            << (
                "input[" +
                std::to_string(dimension) +
                "]"
            )
            << std::setw(18)
            << analytical
            << std::setw(18)
            << numerical
            << std::setw(18)
            << difference
            << '\n';

        expect_near(
            analytical,
            numerical,
            tolerance
        );
    }
}

void test_seeded_initialisation()
{
    using Projection =
        latent::experimental::CliffordProjection;

    Projection a(42);
    Projection b(42);
    Projection c(1337);

    for (
        std::size_t i = 0;
        i < a.weights().size();
        ++i
    ) {
        expect_near(
            a.weights()[i],
            b.weights()[i],
            0.0f
        );
    }

    bool found_difference = false;

    for (
        std::size_t i = 0;
        i < a.weights().size();
        ++i
    ) {
        if (
            std::fabs(
                a.weights()[i] -
                c.weights()[i]
            ) > 1e-7f
        ) {
            found_difference = true;
            break;
        }
    }

    assert(found_difference);

    std::cout
        << "\nSeeded initialisation passed.\n";
}
void test_xavier_range()
{
    using Projection =
        latent::experimental::CliffordProjection;

    Projection projection(42);

    const float limit =
        std::sqrt(
            6.0f /
            (
                static_cast<float>(
                    Projection::SourceDimensions
                ) +
                static_cast<float>(
                    Projection::BladeCount
                )
            )
        );

    float min_weight =
        projection.weights()[0];

    float max_weight =
        projection.weights()[0];

    for (
        const float weight :
        projection.weights()
    ) {
        assert(weight >= -limit);
        assert(weight <= limit);

        min_weight =
            std::min(
                min_weight,
                weight
            );

        max_weight =
            std::max(
                max_weight,
                weight
            );
    }

    std::cout
        << "\n=== Xavier Initialisation Test ===\n"
        << "limit=" << limit << '\n'
        << "min_weight=" << min_weight << '\n'
        << "max_weight=" << max_weight << '\n';
}
void test_adam_training()
{
    using Projection =
        latent::experimental::CliffordProjection;

    std::cout
        << "\n=== Adam Training Test ===\n";

    Projection projection(42);

    latent::nn::Adam optimizer(
        0.01f
    );

    std::vector<float> input(
        Projection::SourceDimensions,
        0.0f
    );

    // Only first eight dimensions are active.
    for (
        std::size_t i = 0;
        i < Projection::BladeCount;
        ++i
    ) {
        input[i] =
            static_cast<float>(i + 1) /
            8.0f;
    }

    const std::array<
        float,
        Projection::BladeCount
    > target{
        0.25f,
        -0.50f,
        0.75f,
        -1.00f,
        0.50f,
        0.25f,
        -0.25f,
        1.00f
    };

    auto compute_loss =
        [&](const Projection& model)
    {
        const auto output =
            model.forward(input);

        float loss = 0.0f;

        for (
            std::size_t i = 0;
            i < Projection::BladeCount;
            ++i
        ) {
            const float error =
                output.fp32[i] -
                target[i];

            loss +=
                0.5f *
                error * error;
        }

        return loss;
    };

    const float initial_loss =
        compute_loss(projection);

    std::cout
        << "initial_loss="
        << initial_loss
        << '\n';

    constexpr std::size_t steps =
        200;

    for (
        std::size_t step = 0;
        step < steps;
        ++step
    ) {
        const auto output =
            projection.forward(input);

        std::array<
            float,
            Projection::BladeCount
        > grad_output{};

        for (
            std::size_t i = 0;
            i < Projection::BladeCount;
            ++i
        ) {
            grad_output[i] =
                output.fp32[i] -
                target[i];
        }

        const auto backward =
            projection.backward(
                input,
                grad_output
            );

        auto parameters =
            projection.weights_vector();

        std::vector<float> gradients(
            backward.grad_weights.begin(),
            backward.grad_weights.end()
        );

        optimizer.step(
            parameters,
            gradients
        );

        projection.set_weights(
            parameters
        );

        if (
            step == 0 ||
            step == 49 ||
            step == 99 ||
            step == 199
        ) {
            std::cout
                << "step="
                << (step + 1)
                << " loss="
                << compute_loss(projection)
                << '\n';
        }
    }

    const float final_loss =
        compute_loss(projection);

    std::cout
        << "final_loss="
        << final_loss
        << '\n';

    assert(
        final_loss <
        initial_loss
    );

    assert(
        final_loss <
        initial_loss * 0.05f
    );

    assert(
        optimizer.step_count() ==
        steps
    );

    std::cout
        << "Adam optimisation reduced loss successfully.\n";
}
} // namespace


int main()
{
    std::cout
        << std::fixed
        << std::setprecision(8);

    test_known_projection();
    test_fp16_coefficients();
    test_reconstruction();
    test_invalid_input_size();
    test_backward_gradients();
    test_seeded_initialisation();
    test_xavier_range();
    test_adam_training();

    std::cout
        << "\n=================================\n"
        << "All CliffordProjection tests passed.\n"
        << "=================================\n";

    return 0;
}