#include "../../latent/core/FP16.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {

void expect_near(
    float actual,
    float expected,
    float tolerance
) {
    assert(
        std::fabs(actual - expected)
        <= tolerance
    );
}

void test_zero()
{
    const auto fp16 =
        latent::core::float_to_fp16(0.0f);

    assert(fp16 == 0x0000u);

    const float value =
        latent::core::fp16_to_float(fp16);

    assert(value == 0.0f);
}

void test_negative_zero()
{
    const auto fp16 =
        latent::core::float_to_fp16(-0.0f);

    assert(fp16 == 0x8000u);
}

void test_one()
{
    const auto fp16 =
        latent::core::float_to_fp16(1.0f);

    assert(fp16 == 0x3C00u);

    expect_near(
        latent::core::fp16_to_float(fp16),
        1.0f,
        0.0f
    );
}

void test_negative_two()
{
    const auto fp16 =
        latent::core::float_to_fp16(-2.0f);

    assert(fp16 == 0xC000u);

    expect_near(
        latent::core::fp16_to_float(fp16),
        -2.0f,
        0.0f
    );
}

void test_half()
{
    const auto fp16 =
        latent::core::float_to_fp16(0.5f);

    assert(fp16 == 0x3800u);

    expect_near(
        latent::core::fp16_to_float(fp16),
        0.5f,
        0.0f
    );
}

void test_max_finite()
{
    const float max_fp16 = 65504.0f;

    const auto fp16 =
        latent::core::float_to_fp16(
            max_fp16
        );

    expect_near(
        latent::core::fp16_to_float(fp16),
        max_fp16,
        1.0f
    );
}

void test_infinity()
{
    const auto positive =
        latent::core::float_to_fp16(
            std::numeric_limits<float>::infinity()
        );

    const auto negative =
        latent::core::float_to_fp16(
            -std::numeric_limits<float>::infinity()
        );

    assert(positive == 0x7C00u);
    assert(negative == 0xFC00u);

    assert(
        std::isinf(
            latent::core::fp16_to_float(
                positive
            )
        )
    );
}

void test_nan()
{
    const auto fp16 =
        latent::core::float_to_fp16(
            std::numeric_limits<float>::quiet_NaN()
        );

    const float value =
        latent::core::fp16_to_float(fp16);

    assert(std::isnan(value));
}

void test_round_trip()
{
    const float values[] = {
        -10.5f,
        -1.25f,
        -0.125f,
        0.125f,
        1.25f,
        3.14159f,
        10.5f,
        100.0f
    };

    for (float value : values) {
        const auto fp16 =
            latent::core::float_to_fp16(
                value
            );

        const float restored =
            latent::core::fp16_to_float(
                fp16
            );

        expect_near(
            restored,
            value,
            0.01f
        );
    }
}

} // namespace

int main()
{
    test_zero();
    test_negative_zero();
    test_one();
    test_negative_two();
    test_half();
    test_max_finite();
    test_infinity();
    test_nan();
    test_round_trip();

    std::cout
        << "FP16 tests passed.\n";

    return 0;
}