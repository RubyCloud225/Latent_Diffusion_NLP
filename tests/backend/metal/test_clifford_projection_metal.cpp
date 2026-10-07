#include "../../../latent/backend/metal/CliffordProjectionMetal.hpp"
#include "../../../latent/core/FP16.hpp"
#include "../../../latent/experimental/CliffordProjection.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

using Projection =
    latent::experimental::CliffordProjection;

Projection::Matrix
make_identity_projection()
{
    Projection::Matrix matrix{};

    for (
        std::size_t blade = 0;
        blade < Projection::BladeCount;
        ++blade
    ) {
        matrix[
            blade *
            Projection::SourceDimensions +
            blade
        ] = 1.0f;
    }

    return matrix;
}


std::vector<float>
make_test_input()
{
    std::vector<float> input(
        Projection::SourceDimensions
    );

    for (
        std::size_t i = 0;
        i < input.size();
        ++i
    ) {
        input[i] =
            static_cast<float>(
                i + 1
            );
    }

    return input;
}


void expect_near(
    float actual,
    float expected,
    float tolerance
)
{
    const float difference =
        std::fabs(
            actual -
            expected
        );

    if (difference > tolerance) {
        std::cerr
            << "FAIL: actual="
            << actual
            << " expected="
            << expected
            << " difference="
            << difference
            << " tolerance="
            << tolerance
            << '\n';
    }

    assert(
        difference <= tolerance
    );
}


void test_known_answer()
{
    std::cout
        << "\n=== Metal Known Answer Test ===\n";

    const auto matrix =
        make_identity_projection();

    const auto input =
        make_test_input();

    const auto gpu =
        latent::backend::metal::
            CliffordProjectionMetal::forward(
                input,
                matrix
            );

    for (
        std::size_t i = 0;
        i < Projection::BladeCount;
        ++i
    ) {
        const float expected =
            static_cast<float>(
                i + 1
            );

        const float gpu_fp16 =
            latent::core::fp16_to_float(
                gpu.fp16[i]
            );

        std::cout
            << "blade[" << i << "] "
            << "gpu_fp32="
            << gpu.fp32[i]
            << " gpu_fp16="
            << gpu_fp16
            << " expected="
            << expected
            << '\n';

        expect_near(
            gpu.fp32[i],
            expected,
            1e-5f
        );

        expect_near(
            gpu_fp16,
            expected,
            1e-3f
        );
    }
}


void test_cpu_gpu_parity()
{
    std::cout
        << "\n=== CPU / Metal Parity Test ===\n";

    Projection cpuProjection(42);

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
            static_cast<float>(
                i + 1
            );
    }

    const auto cpu =
        cpuProjection.forward(
            input
        );

    const auto gpu =
        latent::backend::metal::
            CliffordProjectionMetal::forward(
                input,
                cpuProjection.weights()
            );

    std::cout
        << std::left
        << std::setw(10)
        << "Blade"
        << std::setw(18)
        << "CPU"
        << std::setw(18)
        << "Metal"
        << std::setw(18)
        << "Difference"
        << '\n';

    std::cout
        << std::string(64, '-')
        << '\n';

    for (
        std::size_t i = 0;
        i < Projection::BladeCount;
        ++i
    ) {
        const float difference =
            std::fabs(
                cpu.fp32[i] -
                gpu.fp32[i]
            );

        std::cout
            << std::left
            << std::setw(10)
            << i
            << std::setw(18)
            << cpu.fp32[i]
            << std::setw(18)
            << gpu.fp32[i]
            << std::setw(18)
            << difference
            << '\n';

        expect_near(
            cpu.fp32[i],
            gpu.fp32[i],
            1e-5f
        );
    }
}
void test_backward_parity()
{
    std::cout
        << "\n=== CPU / Metal Backward Parity Test ===\n";

    Projection projection(42);

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

    const std::array<
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

    const auto cpu =
        projection.backward(
            input,
            grad_output
        );

    const auto gpu =
        latent::backend::metal::
            CliffordProjectionMetal::backward(
                input,
                grad_output,
                projection.weights()
            );

    std::cout
        << "\nInput gradients\n";

    std::cout
        << std::left
        << std::setw(12) << "Index"
        << std::setw(18) << "CPU"
        << std::setw(18) << "Metal"
        << std::setw(18) << "Difference"
        << '\n';

    std::cout
        << std::string(66, '-')
        << '\n';

    for (
        std::size_t i = 0;
        i < Projection::SourceDimensions;
        ++i
    ) {
        const float difference =
            std::fabs(
                cpu.grad_input[i] -
                gpu.grad_input[i]
            );

        if (
            i < 8 ||
            i == 31 ||
            i == 63
        ) {
            std::cout
                << std::left
                << std::setw(12)
                << i
                << std::setw(18)
                << cpu.grad_input[i]
                << std::setw(18)
                << gpu.grad_input[i]
                << std::setw(18)
                << difference
                << '\n';
        }

        expect_near(
            cpu.grad_input[i],
            gpu.grad_input[i],
            1e-5f
        );
    }

    std::cout
        << "\nWeight gradients\n";

    std::cout
        << std::left
        << std::setw(14) << "Parameter"
        << std::setw(18) << "CPU"
        << std::setw(18) << "Metal"
        << std::setw(18) << "Difference"
        << '\n';

    std::cout
        << std::string(68, '-')
        << '\n';

    const std::size_t test_indices[] = {
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
        test_indices
    ) {
        const float difference =
            std::fabs(
                cpu.grad_weights[index] -
                gpu.grad_weights[index]
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
            << cpu.grad_weights[index]
            << std::setw(18)
            << gpu.grad_weights[index]
            << std::setw(18)
            << difference
            << '\n';

        expect_near(
            cpu.grad_weights[index],
            gpu.grad_weights[index],
            1e-5f
        );
    }
}

} // namespace


int main()
{
    std::cout
        << std::fixed
        << std::setprecision(8);

    test_known_answer();
    test_cpu_gpu_parity();
    test_backward_parity();
    std::cout
        << "\n=================================\n"
        << "All Metal Clifford tests passed.\n"
        << "=================================\n";

    return 0;
}