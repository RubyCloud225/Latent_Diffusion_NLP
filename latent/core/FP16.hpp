#pragma once

#include <cstdint>
#include <cstring>

namespace latent::core {

/**
 * @brief Convert IEEE-754 single precision (FP32) to binary16 (FP16).
 *
 * @param value FP32 value.
 * @return Raw 16-bit IEEE-754 binary16 representation.
 */
inline std::uint16_t float_to_fp16(float value) noexcept
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const std::uint32_t sign =
        (bits >> 16) & 0x8000u;

    const std::uint32_t exponent =
        (bits >> 23) & 0xFFu;

    std::uint32_t mantissa =
        bits & 0x7FFFFFu;

    // NaN / infinity
    if (exponent == 0xFFu) {
        if (mantissa == 0) {
            return static_cast<std::uint16_t>(
                sign | 0x7C00u
            );
        }

        // Preserve NaN payload where practical.
        std::uint16_t payload =
            static_cast<std::uint16_t>(
                mantissa >> 13
            );

        if (payload == 0) {
            payload = 1;
        }

        return static_cast<std::uint16_t>(
            sign |
            0x7C00u |
            payload
        );
    }

    const int fp16_exponent =
        static_cast<int>(exponent) - 127 + 15;

    // Underflow to signed zero.
    if (fp16_exponent < -10) {
        return static_cast<std::uint16_t>(sign);
    }

    // Subnormal FP16.
    if (fp16_exponent <= 0) {
        mantissa |= 0x800000u;

        const int shift =
            14 - fp16_exponent;

        std::uint32_t half_mantissa =
            mantissa >> shift;

        // Round to nearest.
        const std::uint32_t round_bit =
            (mantissa >> (shift - 1)) & 1u;

        half_mantissa += round_bit;

        return static_cast<std::uint16_t>(
            sign | half_mantissa
        );
    }

    // Overflow to infinity.
    if (fp16_exponent >= 31) {
        return static_cast<std::uint16_t>(
            sign | 0x7C00u
        );
    }

    std::uint32_t half_exponent =
        static_cast<std::uint32_t>(
            fp16_exponent
        ) << 10;

    std::uint32_t half_mantissa =
        mantissa >> 13;

    // Round to nearest.
    if (mantissa & 0x1000u) {
        ++half_mantissa;

        // Mantissa overflow after rounding.
        if (half_mantissa & 0x0400u) {
            half_mantissa = 0;
            half_exponent += 0x0400u;

            if (half_exponent >= 0x7C00u) {
                half_exponent = 0x7C00u;
            }
        }
    }

    return static_cast<std::uint16_t>(
        sign |
        half_exponent |
        (half_mantissa & 0x03FFu)
    );
}


/**
 * @brief Convert raw IEEE-754 binary16 bits to FP32.
 *
 * @param value Raw FP16 bit representation.
 * @return Equivalent FP32 value.
 */
inline float fp16_to_float(
    std::uint16_t value
) noexcept
{
    const std::uint32_t sign =
        static_cast<std::uint32_t>(
            value & 0x8000u
        ) << 16;

    std::uint32_t exponent =
        (value >> 10) & 0x1Fu;

    std::uint32_t mantissa =
        value & 0x03FFu;

    std::uint32_t bits = 0;

    // Zero or subnormal.
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        }
        else {
            int fp32_exponent = -14;

            while (
                (mantissa & 0x0400u) == 0
            ) {
                mantissa <<= 1;
                --fp32_exponent;
            }

            mantissa &= 0x03FFu;

            bits =
                sign |
                (
                    static_cast<std::uint32_t>(
                        fp32_exponent + 127
                    ) << 23
                ) |
                (mantissa << 13);
        }
    }

    // Infinity or NaN.
    else if (exponent == 0x1Fu) {
        bits =
            sign |
            0x7F800000u |
            (mantissa << 13);
    }

    // Normal FP16.
    else {
        const std::uint32_t fp32_exponent =
            exponent - 15u + 127u;

        bits =
            sign |
            (fp32_exponent << 23) |
            (mantissa << 13);
    }

    float result = 0.0f;
    std::memcpy(
        &result,
        &bits,
        sizeof(result)
    );

    return result;
}

} // namespace latent::core