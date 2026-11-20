#include "latent/core/Random.hpp"
#include "latent/diffusion/GaussianDiffusion.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using latent::core::RandomGenerator;
using latent::core::Tensor;
using latent::diffusion::GaussianDiffusion;
using latent::diffusion::NoiseSchedule;

namespace {

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

void near(double actual, double expected, double tol)
{
    require(std::isfinite(actual) && std::abs(actual - expected) <= tol,
            "Numerical mismatch");
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

void test_known_answer()
{
    // One timestep, beta=0.25, alpha_bar=0.75.
    NoiseSchedule schedule({1, 0.25, 0.25});
    GaussianDiffusion diffusion(schedule);
    Tensor z0({2});
    z0[0] = 2.0;
    z0[1] = -4.0;
    Tensor noise({2});
    noise[0] = 1.0;
    noise[1] = -2.0;

    const Tensor result = diffusion.q_sample(z0, 0, noise);
    near(result[0], std::sqrt(0.75) * 2.0 + 0.5, 1e-12);
    near(result[1], std::sqrt(0.75) * -4.0 - 1.0, 1e-12);
}

void test_zero_noise()
{
    NoiseSchedule schedule({10, 0.001, 0.02});
    GaussianDiffusion diffusion(schedule);
    Tensor z0({3});
    z0.fill(2.0);
    Tensor noise({3});
    noise.fill(0.0);

    const Tensor result = diffusion.q_sample(z0, 5, noise);
    for (std::size_t i = 0; i < result.size(); ++i) {
        near(result[i], schedule.sqrt_alpha_bar(5) * 2.0, 1e-12);
    }
}

void test_zero_input()
{
    NoiseSchedule schedule({10, 0.001, 0.02});
    GaussianDiffusion diffusion(schedule);
    Tensor z0({3});
    z0.fill(0.0);
    Tensor noise({3});
    noise.fill(1.5);

    const Tensor result = diffusion.q_sample(z0, 5, noise);
    for (std::size_t i = 0; i < result.size(); ++i) {
        near(result[i], schedule.sqrt_one_minus_alpha_bar(5) * 1.5, 1e-12);
    }
}

void test_shape_preservation()
{
    NoiseSchedule schedule({100, 0.0001, 0.02});
    GaussianDiffusion diffusion(schedule);
    Tensor z0({2, 3, 4, 5});
    z0.fill(0.5);
    Tensor noise({2, 3, 4, 5});
    noise.fill(-0.25);

    const Tensor result = diffusion.q_sample(z0, 50, noise);
    require(result.same_shape(z0), "Forward diffusion changed Tensor shape");
    require(result.size() == z0.size(), "Forward diffusion changed Tensor size");
    require(result.all_finite(), "Non-finite output");
}

void test_input_preservation()
{
    NoiseSchedule schedule({10, 0.001, 0.02});
    GaussianDiffusion diffusion(schedule);
    Tensor z0({2});
    z0[0] = 3.0;
    z0[1] = -2.0;
    Tensor noise({2});
    noise[0] = 0.25;
    noise[1] = -0.5;
    const Tensor original = z0;
    const Tensor original_noise = noise;

    (void)diffusion.q_sample(z0, 5, noise);
    for (std::size_t i = 0; i < z0.size(); ++i) {
        require(z0[i] == original[i], "Input Tensor was modified");
        require(noise[i] == original_noise[i], "Noise Tensor was modified");
    }
}

void test_timestep_boundaries()
{
    NoiseSchedule schedule({10, 0.001, 0.02});
    GaussianDiffusion diffusion(schedule);
    Tensor z0({4});
    z0.fill(1.0);
    Tensor noise({4});
    noise.fill(0.5);

    require(diffusion.q_sample(z0, 0, noise).all_finite(), "First timestep failed");
    require(diffusion.q_sample(z0, 9, noise).all_finite(), "Last timestep failed");
    expect_throw<std::out_of_range>([&] {
        (void)diffusion.q_sample(z0, 10, noise);
    });
}

void test_shape_mismatch()
{
    NoiseSchedule schedule({10, 0.001, 0.02});
    GaussianDiffusion diffusion(schedule);
    Tensor z0({2, 3});
    Tensor noise({3, 2});
    expect_throw<std::invalid_argument>([&] {
        (void)diffusion.q_sample(z0, 5, noise);
    });
}

void test_non_finite_inputs()
{
    NoiseSchedule schedule({10, 0.001, 0.02});
    GaussianDiffusion diffusion(schedule);
    Tensor z0({3});
    Tensor noise({3});
    z0[0] = std::numeric_limits<double>::quiet_NaN();

    expect_throw<std::invalid_argument>([&] {
        (void)diffusion.q_sample(z0, 5, noise);
    });

    z0[0] = 0.0;
    noise[0] = std::numeric_limits<double>::infinity();
    expect_throw<std::invalid_argument>([&] {
        (void)diffusion.q_sample(z0, 5, noise);
    });
}

void test_output_overflow()
{
    NoiseSchedule schedule({1, 0.25, 0.25});
    GaussianDiffusion diffusion(schedule);
    Tensor z0({1});
    Tensor noise({1});
    z0[0] = std::numeric_limits<double>::max();
    noise[0] = std::numeric_limits<double>::max();

    expect_throw<std::overflow_error>([&] {
        (void)diffusion.q_sample(z0, 0, noise);
    });
}

void test_deterministic_repeat()
{
    NoiseSchedule schedule({100, 0.0001, 0.02});
    GaussianDiffusion diffusion(schedule);
    Tensor z0({3, 8, 8});
    z0.fill(0.5);
    RandomGenerator rng(42);
    const Tensor noise = rng.gaussian_tensor(z0.shape());
    const Tensor first = diffusion.q_sample(z0, 50, noise);
    const Tensor second = diffusion.q_sample(z0, 50, noise);

    for (std::size_t i = 0; i < first.size(); ++i) {
        require(first[i] == second[i], "Forward diffusion was not deterministic");
    }
}

} // namespace

int main()
{
    try {
        test_known_answer();
        test_zero_noise();
        test_zero_input();
        test_shape_preservation();
        test_input_preservation();
        test_timestep_boundaries();
        test_shape_mismatch();
        test_non_finite_inputs();
        test_output_overflow();
        test_deterministic_repeat();

        std::cout << "[PASS] P0.7 Forward Diffusion tests\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] P0.7 Forward Diffusion: " << error.what() << '\n';
        return 1;
    }
}
