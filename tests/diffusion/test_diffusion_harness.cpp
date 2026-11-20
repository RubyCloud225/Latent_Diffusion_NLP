// P0.3 test-harness reference fixtures.
//
// This executable validates the mathematical fixture and CTest wiring only.
// It DOES NOT exercise a production diffusion implementation, which is
// introduced in P0.6-P0.9. Replace/add production API tests at those tickets.
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require_close(double actual, double expected, double tolerance, const char* name) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(name);
    }
}
}

int main() {
    try {
        // At cumulative alpha=1, q_sample(x0, t, epsilon) is exactly x0.
        // At cumulative alpha=0, it is exactly epsilon.
        const std::vector<double> input{1.0, -2.0, 0.5};
        const std::vector<double> noise{0.25, 0.75, -0.5};
        for (std::size_t i = 0; i < input.size(); ++i) {
            const double clean = std::sqrt(1.0) * input[i] + std::sqrt(0.0) * noise[i];
            const double noisy = std::sqrt(0.0) * input[i] + std::sqrt(1.0) * noise[i];
            require_close(clean, input[i], 0.0, "clean reference fixture failed");
            require_close(noisy, noise[i], 0.0, "noise reference fixture failed");
        }
        std::cout << "[PASS] P0.3 diffusion reference fixtures (harness only)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
