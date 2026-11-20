#include "latent/diffusion/GaussianDiffusion.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using latent::core::Tensor;
using latent::diffusion::GaussianDiffusion;
using latent::diffusion::NoiseSchedule;

namespace {
void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}
void near(double actual, double expected, double tolerance)
{
    require(std::isfinite(actual) &&
            std::abs(actual - expected) <= tolerance,
            "Numerical mismatch");
}
template<typename Exception, typename Function>
void expect_throw(Function fn)
{
    bool caught = false;
    try { fn(); }
    catch (const Exception&) { caught = true; }
    require(caught, "Expected exception not thrown");
}
void test_posterior_coefficients()
{
    NoiseSchedule schedule({2, 0.1, 0.2});
    GaussianDiffusion diffusion(schedule);
    const auto c = diffusion.reverse_coefficients(1);
    // alpha_bar[0]=0.9; alpha_bar[1]=0.72.
    near(c.posterior_variance, 0.2 * 0.1 / 0.28, 1e-12);
    near(c.mean_coefficient_x0,
         0.2 * std::sqrt(0.9) / 0.28, 1e-12);
    near(c.mean_coefficient_xt,
         0.1 * std::sqrt(0.8) / 0.28, 1e-12);
    near(diffusion.posterior_variance(1), 0.2 * 0.1 / 0.28, 1e-12);
}
void test_known_answer_reverse_mean()
{
    NoiseSchedule schedule({2, 0.1, 0.2});
    GaussianDiffusion diffusion(schedule);
    Tensor zt({2});
    zt[0] = 1.0; zt[1] = -2.0;
    Tensor eps({2});
    eps[0] = 0.5; eps[1] = -0.25;
    const Tensor mean = diffusion.reverse_mean(zt, 1, eps);
    const double c1 = 0.2 * std::sqrt(0.9) / 0.28;
    const double c2 = 0.1 * std::sqrt(0.8) / 0.28;
    const double x0a = (1.0 - std::sqrt(0.28) * 0.5) /
                       std::sqrt(0.72);
    const double x0b = (-2.0 - std::sqrt(0.28) * -0.25) /
                       std::sqrt(0.72);
    near(mean[0], c1 * x0a + c2, 1e-12);
    near(mean[1], c1 * x0b - 2.0 * c2, 1e-12);
}
void test_final_reverse_step()
{
    NoiseSchedule schedule({10, 0.001, 0.02});
    GaussianDiffusion diffusion(schedule);
    Tensor zt({3}); zt.fill(0.75);
    Tensor eps({3}); eps.fill(0.25);
    const Tensor mean = diffusion.reverse_mean(zt, 0, eps);
    const Tensor x0 = diffusion.predict_x0(zt, 0, eps);
    near(diffusion.posterior_variance(0), 0.0, 0.0);
    const auto c = diffusion.reverse_coefficients(0);
    near(c.mean_coefficient_x0, 1.0, 1e-12);
    near(c.mean_coefficient_xt, 0.0, 0.0);
    for (std::size_t i = 0; i < mean.size(); ++i)
        near(mean[i], x0[i], 1e-12);
}
void test_clean_latent_recovery()
{
    NoiseSchedule schedule({100, 0.0001, 0.02});
    GaussianDiffusion diffusion(schedule);
    Tensor clean({2, 3, 4});
    Tensor noise({2, 3, 4});
    for (std::size_t i = 0; i < clean.size(); ++i) {
        clean[i] = 0.1 * static_cast<double>(i);
        noise[i] = -0.02 * static_cast<double>(i);
    }
    const Tensor noisy = diffusion.q_sample(clean, 50, noise);
    const Tensor recovered = diffusion.predict_x0(noisy, 50, noise);
    for (std::size_t i = 0; i < clean.size(); ++i)
        near(recovered[i], clean[i], 1e-12);
}
void test_shape_and_input_preservation()
{
    NoiseSchedule schedule({10, 0.001, 0.02});
    GaussianDiffusion diffusion(schedule);
    Tensor zt({2, 3, 4, 5}); zt.fill(0.4);
    Tensor eps({2, 3, 4, 5}); eps.fill(-0.2);
    const Tensor old_zt = zt;
    const Tensor old_eps = eps;
    const Tensor result = diffusion.reverse_mean(zt, 5, eps);
    require(result.same_shape(zt), "Shape changed");
    require(result.all_finite(), "Non-finite reverse mean");
    for (std::size_t i = 0; i < zt.size(); ++i) {
        require(zt[i] == old_zt[i], "zt mutated");
        require(eps[i] == old_eps[i], "epsilon mutated");
    }
}
void test_deterministic_reverse_mean()
{
    NoiseSchedule schedule({100, 0.0001, 0.02});
    GaussianDiffusion diffusion(schedule);
    Tensor zt({8}); zt.fill(0.5);
    Tensor eps({8}); eps.fill(0.1);
    const Tensor a = diffusion.reverse_mean(zt, 40, eps);
    const Tensor b = diffusion.reverse_mean(zt, 40, eps);
    for (std::size_t i = 0; i < a.size(); ++i)
        require(a[i] == b[i], "Reverse mean differs");
}
void test_invalid_shapes_and_timesteps()
{
    NoiseSchedule schedule({10, 0.001, 0.02});
    GaussianDiffusion diffusion(schedule);
    Tensor zt({2, 3});
    Tensor wrong({3, 2});
    Tensor eps({2, 3});
    expect_throw<std::invalid_argument>([&] {
        (void)diffusion.reverse_mean(zt, 5, wrong);
    });
    expect_throw<std::out_of_range>([&] {
        (void)diffusion.reverse_mean(zt, 10, eps);
    });
    expect_throw<std::out_of_range>([&] {
        (void)diffusion.posterior_variance(10);
    });
}
void test_non_finite_inputs()
{
    NoiseSchedule schedule({10, 0.001, 0.02});
    GaussianDiffusion diffusion(schedule);
    Tensor zt({2});
    Tensor eps({2});
    eps[0] = std::numeric_limits<double>::quiet_NaN();
    expect_throw<std::invalid_argument>([&] {
        (void)diffusion.reverse_mean(zt, 5, eps);
    });
    eps[0] = 0.0;
    zt[0] = std::numeric_limits<double>::infinity();
    expect_throw<std::invalid_argument>([&] {
        (void)diffusion.predict_x0(zt, 5, eps);
    });
}
void test_variance_range()
{
    NoiseSchedule schedule({1000, 0.0001, 0.02});
    GaussianDiffusion diffusion(schedule);
    for (std::size_t t = 0; t < schedule.timesteps(); ++t) {
        const double variance = diffusion.posterior_variance(t);
        require(std::isfinite(variance), "Non-finite posterior variance");
        require(variance >= 0.0, "Negative posterior variance");
        require(variance <= schedule.beta(t) + 1e-15,
                "Posterior variance exceeds beta");
    }
}
} // namespace

int main()
{
    try {
        test_posterior_coefficients();
        test_known_answer_reverse_mean();
        test_final_reverse_step();
        test_clean_latent_recovery();
        test_shape_and_input_preservation();
        test_deterministic_reverse_mean();
        test_invalid_shapes_and_timesteps();
        test_non_finite_inputs();
        test_variance_range();
        std::cout << "[PASS] P0.8 Reverse Diffusion Mathematics\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] P0.8: " << error.what() << '\n';
        return 1;
    }
}
