#include "latent/nn/training/TrainingPolicy.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using latent::nn::training::GradientMode;
using latent::nn::training::Objective;
using latent::nn::training::TrainingPolicy;

namespace {

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

template<typename Exception, typename Function>
void expect_throw(Function fn)
{
    bool caught = false;
    try {
        fn();
    } catch (const Exception&) {
        caught = true;
    }
    require(caught, "Expected exception not thrown");
}

TrainingPolicy reference_policy()
{
    TrainingPolicy p;
    p.objective = Objective::EpsilonMSE;
    p.dataset_manifest = "dataset-v1";
    p.train_split = "train-v1";
    p.validation_split = "validation-v1";
    p.test_split = "test-v1";
    p.code_revision = "test-commit";
    p.gradient_mode = GradientMode::MetricRadial;
    p.metric_id = "equal_block_rms_v1";
    return p;
}

void test_selected_configuration()
{
    const auto p = reference_policy();
    p.validate();
    require(p.batch_size == 16, "Wrong micro-batch");
    require(p.accumulation_steps == 8, "Wrong accumulation");
    require(p.effective_batch_size() == 128, "Wrong effective batch");
    require(p.learning_rate == 0.001, "Wrong Adam learning rate");
    require(p.gradient_mode == GradientMode::MetricRadial,
            "Wrong metric mode");
    require(!p.execution_enabled, "Training execution unexpectedly enabled");
}

void test_round_trip_and_quoted_strings()
{
    auto p = reference_policy();
    p.dataset_manifest = "manifest = alpha \"quoted\" path \\ dataset";
    p.code_revision = "abc/123";
    p.learning_rate = std::nextafter(0.001, 1.0);
    const auto canonical = p.serialize();
    require(canonical == p.serialize(), "Nondeterministic encoding");
    const auto decoded = TrainingPolicy::parse(canonical);
    require(decoded.serialize() == canonical, "FP64 or string round-trip loss");
    require(decoded.dataset_manifest == p.dataset_manifest, "Escaped string lost");
    require(decoded.learning_rate == p.learning_rate, "FP64 precision lost");
    require(decoded.objective == Objective::EpsilonMSE, "Objective lost");

    p.objective = Objective::ReconstructionMSE;
    p.gradient_mode = GradientMode::None;
    p.metric_id.clear();
    require(TrainingPolicy::parse(p.serialize()).objective ==
            Objective::ReconstructionMSE, "Autoencoder mode lost");
}

void test_euclidean_and_none()
{
    auto p = reference_policy();
    p.gradient_mode = GradientMode::EuclideanGlobal;
    p.metric_id.clear();
    require(TrainingPolicy::parse(p.serialize()).gradient_mode ==
            GradientMode::EuclideanGlobal, "Euclidean mode lost");
    p.gradient_mode = GradientMode::None;
    require(TrainingPolicy::parse(p.serialize()).gradient_mode ==
            GradientMode::None, "Unclipped mode lost");
}

void test_invalid_policy()
{
    auto p = reference_policy();
    p.metric_id = "unsupported";
    expect_throw<std::invalid_argument>([&]{ p.validate(); });
    p = reference_policy();
    p.metric_id.clear();
    expect_throw<std::invalid_argument>([&]{ p.validate(); });
    p = reference_policy();
    p.gradient_mode = GradientMode::None;
    expect_throw<std::invalid_argument>([&]{ p.validate(); });
    p = reference_policy();
    p.batch_size = 0;
    expect_throw<std::invalid_argument>([&]{ p.validate(); });
    p = reference_policy();
    p.accumulation_steps = 0;
    expect_throw<std::invalid_argument>([&]{ p.validate(); });
    p = reference_policy();
    p.train_split = p.validation_split;
    expect_throw<std::invalid_argument>([&]{ p.validate(); });
    p = reference_policy();
    p.dataset_manifest = "path\ninjection";
    expect_throw<std::invalid_argument>([&]{ p.validate(); });
    p = reference_policy();
    p.learning_rate = std::numeric_limits<double>::quiet_NaN();
    expect_throw<std::invalid_argument>([&]{ p.validate(); });
    p = reference_policy();
    p.beta1 = 1.0;
    expect_throw<std::invalid_argument>([&]{ p.validate(); });
    p = reference_policy();
    p.beta2 = -0.1;
    expect_throw<std::invalid_argument>([&]{ p.validate(); });
    p = reference_policy();
    p.epsilon = std::numeric_limits<double>::infinity();
    expect_throw<std::invalid_argument>([&]{ p.validate(); });
    p = reference_policy();
    p.max_gradient_norm = 0.0;
    expect_throw<std::invalid_argument>([&]{ p.validate(); });
    p = reference_policy();
    p.gradient_mode = static_cast<GradientMode>(1234);
    expect_throw<std::invalid_argument>([&]{ p.validate(); });
    p = reference_policy();
    p.objective = static_cast<Objective>(1234);
    expect_throw<std::invalid_argument>([&]{ p.validate(); });
    p = reference_policy();
    p.execution_enabled = true;
    expect_throw<std::invalid_argument>([&]{ p.validate(); });
}

void test_effective_batch_overflow()
{
    auto p = reference_policy();
    p.batch_size = std::numeric_limits<std::size_t>::max();
    p.accumulation_steps = 8;
    expect_throw<std::overflow_error>([&]{ p.validate(); });
}

void test_malformed_and_unknown_fields()
{
    const auto good = reference_policy().serialize();
    expect_throw<std::invalid_argument>([&]{
        (void)TrainingPolicy::parse(good + "unexpected=1\n");
    });
    expect_throw<std::invalid_argument>([&]{
        (void)TrainingPolicy::parse(good + "batch_size=16\n");
    });
    expect_throw<std::invalid_argument>([&]{
        (void)TrainingPolicy::parse(good.substr(0, good.find("execution_enabled=")));
    });

    const auto version_position = good.find("schema_version=1");
    require(version_position != std::string::npos, "Missing version fixture");
    auto wrong_version = good;
    wrong_version.replace(version_position, 16, "schema_version=2");
    expect_throw<std::invalid_argument>([&]{
        (void)TrainingPolicy::parse(wrong_version);
    });

    auto bad_number = good;
    const auto npos = bad_number.find("batch_size=16");
    require(npos != std::string::npos, "Missing batch fixture");
    bad_number.replace(npos, 13, "batch_size=-1");
    expect_throw<std::invalid_argument>([&]{
        (void)TrainingPolicy::parse(bad_number);
    });

    auto bad_bool = good;
    const auto bpos = bad_bool.find("execution_enabled=false");
    require(bpos != std::string::npos, "Missing boolean fixture");
    bad_bool.replace(bpos, 23, "execution_enabled=yes");
    expect_throw<std::invalid_argument>([&]{
        (void)TrainingPolicy::parse(bad_bool);
    });

    auto nonfinite = good;
    const auto lrpos = nonfinite.find("learning_rate=");
    require(lrpos != std::string::npos, "Missing FP64 fixture");
    const auto lrend = nonfinite.find('\n', lrpos);
    nonfinite.replace(lrpos, lrend - lrpos, "learning_rate=nan");
    expect_throw<std::invalid_argument>([&]{
        (void)TrainingPolicy::parse(nonfinite);
    });

    auto unknown_instead_of_known = good;
    const auto cpos = unknown_instead_of_known.find("code_revision=");
    require(cpos != std::string::npos, "Missing code revision");
    unknown_instead_of_known.replace(cpos, 13, "code_revisioX=");
    expect_throw<std::invalid_argument>([&]{
        (void)TrainingPolicy::parse(unknown_instead_of_known);
    });

    expect_throw<std::invalid_argument>([&]{
        (void)TrainingPolicy::parse("schema_version=1\n\n");
    });
}

void test_size_and_uint_limits()
{
    auto p = reference_policy();
    p.seed = std::numeric_limits<std::uint64_t>::max();
    require(TrainingPolicy::parse(p.serialize()).seed == p.seed,
            "Lost maximum uint64 seed");

    auto overflow = p.serialize();
    const auto start = overflow.find("seed=");
    const auto end = overflow.find('\n', start);
    overflow.replace(start, end - start,
                     "seed=184467440737095516160000");
    expect_throw<std::overflow_error>([&]{
        (void)TrainingPolicy::parse(overflow);
    });
}

} // namespace

int main()
{
    try {
        test_selected_configuration();
        test_round_trip_and_quoted_strings();
        test_euclidean_and_none();
        test_invalid_policy();
        test_effective_batch_overflow();
        test_malformed_and_unknown_fields();
        test_size_and_uint_limits();
        std::cout << "[PASS] NN.1.07A Training Policy\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] NN.1.07A: " << error.what() << '\n';
        return 1;
    }
}
