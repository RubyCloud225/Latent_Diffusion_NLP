
#include "latent/diffusion/NoiseSchedule.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using latent::diffusion::NoiseSchedule;
using latent::diffusion::ScheduleConfig;

namespace {

void require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(double actual, double expected, double tol)
{
    require(
        std::isfinite(actual) &&
        std::abs(actual - expected) <= tol,
        "Numerical mismatch"
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

    require(caught, "Expected exception not thrown");
}

void test_linear_schedule()
{
    NoiseSchedule schedule({3, 0.1, 0.3});

    require(schedule.timesteps() == 3,
            "Wrong timestep count");

    near(schedule.beta(0), 0.1, 1e-12);
    near(schedule.beta(1), 0.2, 1e-12);
    near(schedule.beta(2), 0.3, 1e-12);

    near(schedule.alpha(0), 0.9, 1e-12);
    near(schedule.alpha(1), 0.8, 1e-12);
    near(schedule.alpha(2), 0.7, 1e-12);

    near(schedule.alpha_bar(0), 0.9, 1e-12);
    near(schedule.alpha_bar(1), 0.72, 1e-12);
    near(schedule.alpha_bar(2), 0.504, 1e-12);
}

void test_square_roots()
{
    NoiseSchedule schedule({3, 0.1, 0.3});

    for (std::size_t t = 0; t < 3; ++t) {
        near(
            schedule.sqrt_alpha_bar(t),
            std::sqrt(schedule.alpha_bar(t)),
            1e-12
        );

        near(
            schedule.sqrt_one_minus_alpha_bar(t),
            std::sqrt(1.0 - schedule.alpha_bar(t)),
            1e-12
        );
    }
}

void test_single_timestep()
{
    NoiseSchedule schedule({1, 0.01, 0.02});

    require(schedule.timesteps() == 1,
            "Wrong single-step count");

    near(schedule.beta(0), 0.01, 1e-12);
    near(schedule.alpha_bar(0), 0.99, 1e-12);
}

void test_monotonicity()
{
    NoiseSchedule schedule({1000, 0.0001, 0.02});

    for (std::size_t t = 1;
         t < schedule.timesteps(); ++t) {
        require(
            schedule.beta(t) >= schedule.beta(t - 1),
            "Beta is not monotonic"
        );

        require(
            schedule.alpha_bar(t) <
                schedule.alpha_bar(t - 1),
            "Cumulative alpha is not decreasing"
        );
    }
}

void test_finite_values()
{
    NoiseSchedule schedule({1000, 0.0001, 0.02});

    for (std::size_t t = 0;
         t < schedule.timesteps(); ++t) {
        require(std::isfinite(schedule.beta(t)),
                "Non-finite beta");

        require(std::isfinite(schedule.alpha(t)),
                "Non-finite alpha");

        require(std::isfinite(schedule.alpha_bar(t)),
                "Non-finite alpha_bar");

        require(
            std::isfinite(schedule.sqrt_alpha_bar(t)),
            "Non-finite square root"
        );

        require(
            std::isfinite(
                schedule.sqrt_one_minus_alpha_bar(t)
            ),
            "Non-finite noise square root"
        );
    }
}

void test_invalid_configs()
{
    expect_throw<std::invalid_argument>([] {
        NoiseSchedule schedule({0, 0.001, 0.02});
    });

    expect_throw<std::invalid_argument>([] {
        NoiseSchedule schedule({10, 0.0, 0.02});
    });

    expect_throw<std::invalid_argument>([] {
        NoiseSchedule schedule({10, 0.2, 0.1});
    });

    expect_throw<std::invalid_argument>([] {
        NoiseSchedule schedule({10, 0.001, 1.0});
    });

    expect_throw<std::invalid_argument>([] {
        NoiseSchedule schedule({
            10,
            std::numeric_limits<double>::quiet_NaN(),
            0.02
        });
    });
}

void test_timestep_bounds()
{
    NoiseSchedule schedule({10, 0.001, 0.02});

    expect_throw<std::out_of_range>([&] {
        (void)schedule.beta(10);
    });

    expect_throw<std::out_of_range>([&] {
        (void)schedule.alpha_bar(100);
    });
}

void test_deterministic_repeat()
{
    NoiseSchedule first({100, 0.0001, 0.02});
    NoiseSchedule second({100, 0.0001, 0.02});

    for (std::size_t t = 0; t < 100; ++t) {
        require(
            first.alpha_bar(t) == second.alpha_bar(t),
            "Schedule was not deterministic"
        );
    }
}

} // namespace

int main()
{
    try {
        test_linear_schedule();
        test_square_roots();
        test_single_timestep();
        test_monotonicity();
        test_finite_values();
        test_invalid_configs();
        test_timestep_bounds();
        test_deterministic_repeat();

        std::cout
            << "[PASS] P0.6 NoiseSchedule tests\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "[FAIL] P0.6 NoiseSchedule: "
            << error.what() << '\n';

        return 1;
    }
}
