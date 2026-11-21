#include "latent/nn/AdamFP64.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using latent::nn::AdamFP64;
using latent::nn::AdamSnapshot;
using latent::nn::Parameter;

namespace {

void check(bool value, const char* text)
{
    if (!value) throw std::runtime_error(text);
}

template<class E, class F>
void rejects(F action)
{
    bool caught = false;
    try { action(); }
    catch (const E&) { caught = true; }
    check(caught, "Expected Adam snapshot rejection missing");
}

void same_snapshot(const AdamSnapshot& a, const AdamSnapshot& b)
{
    check(a.version == b.version && a.step_count == b.step_count &&
          a.learning_rate == b.learning_rate &&
          a.beta1 == b.beta1 && a.beta2 == b.beta2 &&
          a.epsilon == b.epsilon &&
          a.parameter_ids == b.parameter_ids &&
          a.first_moments.size() == b.first_moments.size() &&
          a.second_moments.size() == b.second_moments.size(),
          "Adam snapshot metadata mismatch");

    for (std::size_t k = 0; k < a.first_moments.size(); ++k) {
        for (const auto* pair : {
                std::make_pair(&a.first_moments[k], &b.first_moments[k]),
                std::make_pair(&a.second_moments[k], &b.second_moments[k])}) {
            const auto& x = *pair.first;
            const auto& y = *pair.second;
            check(x.same_shape(y), "Adam snapshot shape changed");
            for (std::size_t i = 0; i < x.size(); ++i) {
                check(x[i] == y[i], "Adam snapshot FP64 state differs");
            }
        }
    }
}

void set_grads(Parameter& a, Parameter& b, int step)
{
    a.gradients()[0] = 0.5 + step * 0.125;
    a.gradients()[1] = -0.25 - step * 0.0625;
    b.gradients()[0] = 0.125 + step * 0.25;
}

void test_round_trip_and_replay()
{
    Parameter w1({2}), b1({1});
    Parameter w2({2}), b2({1});
    w1.values()[0] = w2.values()[0] = 2.0;
    w1.values()[1] = w2.values()[1] = -1.0;
    b1.values()[0] = b2.values()[0] = 0.5;

    const std::vector<std::string> ids = {"encoder.weight", "encoder.bias"};
    AdamFP64 uninterrupted({&w1, &b1}, 0.001, 0.9, 0.999, 1e-8, ids);
    AdamFP64 partial({&w2, &b2}, 0.001, 0.9, 0.999, 1e-8, ids);

    for (int i = 0; i < 10; ++i) {
        set_grads(w1, b1, i);
        uninterrupted.step();
    }

    for (int i = 0; i < 5; ++i) {
        set_grads(w2, b2, i);
        partial.step();
    }
    const AdamSnapshot saved = partial.snapshot();
    check(saved.version == 1 && saved.step_count == 5,
          "Wrong saved state version/step");
    check(saved.parameter_ids == ids, "Stable parameter IDs missing");
    check(saved.first_moments.size() == 2 &&
          saved.second_moments.size() == 2,
          "Incomplete Adam moments");

    // Model weights belong to the later checkpoint format (E.08).
    // Simulate independently restoring model values before Adam state.
    Parameter wr({2}), br({1});
    wr.values() = w2.values();
    br.values() = b2.values();
    AdamFP64 resumed({&wr, &br}, 0.001, 0.9, 0.999, 1e-8, ids);
    resumed.restore(saved);
    same_snapshot(saved, resumed.snapshot());
    check(resumed.step_count() == 5, "Restored step count incorrect");

    for (int i = 5; i < 10; ++i) {
        set_grads(wr, br, i);
        resumed.step();
    }
    check(uninterrupted.step_count() == resumed.step_count(),
          "Resumed optimisation has wrong step count");
    for (std::size_t i = 0; i < w1.values().size(); ++i) {
        check(w1.values()[i] == wr.values()[i],
              "Resumed weights differ from continuous optimisation");
    }
    check(b1.values()[0] == br.values()[0],
          "Resumed bias differs from continuous optimisation");
    same_snapshot(uninterrupted.snapshot(), resumed.snapshot());
}

void test_restore_rejects_corruption_without_mutation()
{
    Parameter weight({2}), bias({1});
    weight.values()[0] = 1.0;
    bias.values()[0] = -1.0;
    weight.gradients()[0] = 0.25;
    bias.gradients()[0] = -0.5;
    AdamFP64 adam({&weight, &bias}, 0.01, 0.8, 0.95, 1e-7,
                  {"weights", "bias"});
    adam.step();
    const AdamSnapshot original = adam.snapshot();
    const double weight_value = weight.values()[0];
    const double bias_value = bias.values()[0];
    const double gradient_value = weight.gradients()[0];

    auto invalid = original;
    invalid.version = 2;
    rejects<std::invalid_argument>([&] { adam.restore(invalid); });
    same_snapshot(original, adam.snapshot());

    invalid = original;
    invalid.learning_rate = 0.001;
    rejects<std::invalid_argument>([&] { adam.restore(invalid); });

    invalid = original;
    invalid.beta1 = 0.9;
    rejects<std::invalid_argument>([&] { adam.restore(invalid); });

    invalid = original;
    invalid.parameter_ids[0] = "other";
    rejects<std::invalid_argument>([&] { adam.restore(invalid); });

    invalid = original;
    std::swap(invalid.parameter_ids[0], invalid.parameter_ids[1]);
    std::swap(invalid.first_moments[0], invalid.first_moments[1]);
    std::swap(invalid.second_moments[0], invalid.second_moments[1]);
    rejects<std::invalid_argument>([&] { adam.restore(invalid); });

    invalid = original;
    invalid.second_moments[1][0] = -0.1;
    rejects<std::invalid_argument>([&] { adam.restore(invalid); });

    invalid = original;
    invalid.first_moments[1][0] =
        std::numeric_limits<double>::quiet_NaN();
    rejects<std::invalid_argument>([&] { adam.restore(invalid); });

    invalid = original;
    invalid.first_moments.pop_back();
    rejects<std::invalid_argument>([&] { adam.restore(invalid); });

    invalid = original;
    invalid.first_moments[0] = latent::core::Tensor({3});
    rejects<std::invalid_argument>([&] { adam.restore(invalid); });

    same_snapshot(original, adam.snapshot());
    check(weight.values()[0] == weight_value &&
          bias.values()[0] == bias_value &&
          weight.gradients()[0] == gradient_value,
          "Rejected restore mutated model values or gradients");
}

void test_registration_identity_and_default_ids()
{
    Parameter a({1}), b({1});
    rejects<std::invalid_argument>([&] {
        AdamFP64 bad({&a, &b}, 0.001, 0.9, 0.999, 1e-8, {"only"});
    });
    rejects<std::invalid_argument>([&] {
        AdamFP64 bad({&a, &b}, 0.001, 0.9, 0.999, 1e-8,
                     {"duplicate", "duplicate"});
    });
    rejects<std::invalid_argument>([&] {
        AdamFP64 bad({&a}, 0.001, 0.9, 0.999, 1e-8, {"\n"});
    });

    AdamFP64 legacy({&a, &b});
    const auto snapshot = legacy.snapshot();
    check(snapshot.parameter_ids ==
          std::vector<std::string>({"position:0", "position:1"}),
          "Default positional IDs changed");
    AdamFP64 second({&a, &b});
    second.restore(snapshot);
    same_snapshot(snapshot, second.snapshot());
}

void test_snapshot_rejects_invalid_live_state()
{
    Parameter p({1});
    AdamFP64 adam({&p});
    p.gradients()[0] = std::numeric_limits<double>::infinity();
    rejects<std::invalid_argument>([&] {
        (void)adam.snapshot();
    });
}

} // namespace

int main()
{
    try {
        test_round_trip_and_replay();
        test_restore_rejects_corruption_without_mutation();
        test_registration_identity_and_default_ids();
        test_snapshot_rejects_invalid_live_state();
        std::cout << "[PASS] NN.1.07E.06 AdamFP64 state snapshot\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] NN.1.07E.06: " << error.what() << '\n';
        return 1;
    }
}
