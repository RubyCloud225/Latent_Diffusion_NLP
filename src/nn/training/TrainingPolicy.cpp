#include "latent/nn/training/TrainingPolicy.hpp"

#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace latent::nn::training {
namespace {

using Fields = std::map<std::string, std::string>;

void require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

bool valid_identifier(const std::string& value)
{
    if (value.empty()) {
        return false;
    }
    for (unsigned char c : value) {
        if (c < 32 || c == 127) {
            return false;
        }
    }
    return true;
}

const char* objective_name(Objective value)
{
    switch (value) {
        case Objective::ReconstructionMSE: return "reconstruction_mse";
        case Objective::EpsilonMSE: return "epsilon_mse";
    }
    throw std::invalid_argument("Unsupported training objective");
}

const char* gradient_name(GradientMode value)
{
    switch (value) {
        case GradientMode::None: return "none";
        case GradientMode::EuclideanGlobal: return "euclidean";
        case GradientMode::MetricRadial: return "metric_radial";
    }
    throw std::invalid_argument("Unsupported gradient mode");
}

std::string quoted(const std::string& value)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::quoted(value);
    return out.str();
}

std::string encoded_double(double value)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<double>::max_digits10)
        << value;
    return out.str();
}

const std::string& field(const Fields& fields, const char* key)
{
    auto found = fields.find(key);
    if (found == fields.end()) {
        throw std::invalid_argument(std::string("Missing policy field: ") + key);
    }
    return found->second;
}

std::string decode_string(const std::string& encoded)
{
    require(encoded.size() >= 2 && encoded.front() == '"' &&
            encoded.back() == '"', "Expected quoted policy string");
    std::istringstream in(encoded);
    in.imbue(std::locale::classic());
    std::string result;
    in >> std::quoted(result);
    require(!in.fail() && in.peek() == std::char_traits<char>::eof(),
            "Invalid quoted policy string");
    return result;
}

std::uint64_t decode_uint(const std::string& encoded)
{
    require(!encoded.empty(), "Empty unsigned integer");
    for (char c : encoded) {
        require(c >= '0' && c <= '9', "Invalid unsigned integer");
    }

    std::size_t consumed = 0;
    unsigned long long number = 0;
    try {
        number = std::stoull(encoded, &consumed, 10);
    } catch (const std::out_of_range&) {
        throw std::overflow_error("Unsigned policy integer overflow");
    }
    require(consumed == encoded.size(), "Malformed unsigned integer");
    if (number > std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("Policy uint64 overflow");
    }
    return static_cast<std::uint64_t>(number);
}

std::size_t decode_size(const std::string& encoded)
{
    const std::uint64_t value = decode_uint(encoded);
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("Policy size_t overflow");
    }
    return static_cast<std::size_t>(value);
}

double decode_double(const std::string& encoded)
{
    require(!encoded.empty(), "Empty FP64 value");
    std::istringstream in(encoded);
    in.imbue(std::locale::classic());
    double value = 0.0;
    in >> std::noskipws >> value;
    require(!in.fail() && in.peek() == std::char_traits<char>::eof(),
            "Malformed or out-of-range FP64 value");
    require(std::isfinite(value), "Non-finite FP64 policy value");
    return value;
}

bool decode_bool(const std::string& encoded)
{
    if (encoded == "true") return true;
    if (encoded == "false") return false;
    throw std::invalid_argument("Expected true or false");
}

Objective decode_objective(const std::string& encoded)
{
    if (encoded == "reconstruction_mse") return Objective::ReconstructionMSE;
    if (encoded == "epsilon_mse") return Objective::EpsilonMSE;
    throw std::invalid_argument("Unsupported training objective");
}

GradientMode decode_gradient(const std::string& encoded)
{
    if (encoded == "none") return GradientMode::None;
    if (encoded == "euclidean") return GradientMode::EuclideanGlobal;
    if (encoded == "metric_radial") return GradientMode::MetricRadial;
    throw std::invalid_argument("Unsupported gradient mode");
}

} // namespace

void TrainingPolicy::validate() const
{
    (void)objective_name(objective);
    (void)gradient_name(gradient_mode);

    require(valid_identifier(dataset_manifest), "Dataset manifest required");
    require(valid_identifier(train_split) &&
            valid_identifier(validation_split) &&
            valid_identifier(test_split), "Dataset split identifiers required");
    require(train_split != validation_split &&
            train_split != test_split &&
            validation_split != test_split, "Dataset split identifiers must differ");
    require(valid_identifier(code_revision), "Code revision required");

    require(batch_size > 0 && accumulation_steps > 0 &&
            max_optimiser_steps > 0, "Invalid batch or optimiser step limit");
    (void)effective_batch_size();

    require(std::isfinite(learning_rate) && learning_rate > 0.0,
            "Invalid Adam learning rate");
    require(std::isfinite(beta1) && beta1 >= 0.0 && beta1 < 1.0 &&
            std::isfinite(beta2) && beta2 >= 0.0 && beta2 < 1.0,
            "Invalid Adam beta");
    require(std::isfinite(epsilon) && epsilon > 0.0, "Invalid Adam epsilon");
    require(std::isfinite(max_gradient_norm) && max_gradient_norm > 0.0,
            "Invalid gradient norm");

    if (gradient_mode == GradientMode::MetricRadial) {
        require(metric_id == "equal_block_rms_v1", "Unsupported metric");
    } else {
        require(metric_id.empty(), "Metric ID requires metric radial mode");
    }

    require(validation_interval > 0 && checkpoint_interval > 0 &&
            max_consecutive_failures > 0, "Invalid monitoring/failure limits");
    require(!execution_enabled, "Training execution is not implemented");
}

std::size_t TrainingPolicy::effective_batch_size() const
{
    if (accumulation_steps != 0 &&
        batch_size > std::numeric_limits<std::size_t>::max() /
                         accumulation_steps) {
        throw std::overflow_error("Effective batch size overflow");
    }
    return batch_size * accumulation_steps;
}

std::string TrainingPolicy::serialize() const
{
    validate();
    std::ostringstream out;
    out.imbue(std::locale::classic());

    out << "schema_version=1\n";
    out << "objective=" << objective_name(objective) << '\n';
    out << "dataset_manifest=" << quoted(dataset_manifest) << '\n';
    out << "train_split=" << quoted(train_split) << '\n';
    out << "validation_split=" << quoted(validation_split) << '\n';
    out << "test_split=" << quoted(test_split) << '\n';
    out << "code_revision=" << quoted(code_revision) << '\n';
    out << "seed=" << seed << '\n';
    out << "batch_size=" << batch_size << '\n';
    out << "accumulation_steps=" << accumulation_steps << '\n';
    out << "max_optimiser_steps=" << max_optimiser_steps << '\n';
    out << "learning_rate=" << encoded_double(learning_rate) << '\n';
    out << "beta1=" << encoded_double(beta1) << '\n';
    out << "beta2=" << encoded_double(beta2) << '\n';
    out << "epsilon=" << encoded_double(epsilon) << '\n';
    out << "gradient_mode=" << gradient_name(gradient_mode) << '\n';
    out << "max_gradient_norm=" << encoded_double(max_gradient_norm) << '\n';
    out << "metric_id=" << quoted(metric_id) << '\n';
    out << "validation_interval=" << validation_interval << '\n';
    out << "checkpoint_interval=" << checkpoint_interval << '\n';
    out << "max_consecutive_failures=" << max_consecutive_failures << '\n';
    out << "record_gradient_norms="
        << (record_gradient_norms ? "true" : "false") << '\n';
    out << "save_optimizer_state="
        << (save_optimizer_state ? "true" : "false") << '\n';
    out << "save_rng_state=" << (save_rng_state ? "true" : "false") << '\n';
    out << "execution_enabled=" << (execution_enabled ? "true" : "false")
        << '\n';
    return out.str();
}

TrainingPolicy TrainingPolicy::parse(const std::string& text)
{
    std::istringstream in(text);
    in.imbue(std::locale::classic());
    Fields fields;
    std::string line;

    while (std::getline(in, line)) {
        require(!line.empty(), "Empty policy record line");
        const auto pos = line.find('=');
        require(pos != std::string::npos && pos > 0,
                "Malformed policy record line");
        const std::string key = line.substr(0, pos);
        const std::string value = line.substr(pos + 1);
        require(fields.emplace(key, value).second, "Duplicate policy field");
    }

    constexpr std::size_t expected_fields = 25;
    require(fields.size() == expected_fields,
            "Missing or unknown policy field");
    require(field(fields, "schema_version") == "1",
            "Unsupported policy schema version");

    TrainingPolicy p;
    p.objective = decode_objective(field(fields, "objective"));
    p.dataset_manifest = decode_string(field(fields, "dataset_manifest"));
    p.train_split = decode_string(field(fields, "train_split"));
    p.validation_split = decode_string(field(fields, "validation_split"));
    p.test_split = decode_string(field(fields, "test_split"));
    p.code_revision = decode_string(field(fields, "code_revision"));
    p.seed = decode_uint(field(fields, "seed"));
    p.batch_size = decode_size(field(fields, "batch_size"));
    p.accumulation_steps = decode_size(field(fields, "accumulation_steps"));
    p.max_optimiser_steps = decode_size(field(fields, "max_optimiser_steps"));
    p.learning_rate = decode_double(field(fields, "learning_rate"));
    p.beta1 = decode_double(field(fields, "beta1"));
    p.beta2 = decode_double(field(fields, "beta2"));
    p.epsilon = decode_double(field(fields, "epsilon"));
    p.gradient_mode = decode_gradient(field(fields, "gradient_mode"));
    p.max_gradient_norm = decode_double(field(fields, "max_gradient_norm"));
    p.metric_id = decode_string(field(fields, "metric_id"));
    p.validation_interval = decode_size(field(fields, "validation_interval"));
    p.checkpoint_interval = decode_size(field(fields, "checkpoint_interval"));
    p.max_consecutive_failures = decode_size(
        field(fields, "max_consecutive_failures"));
    p.record_gradient_norms = decode_bool(field(fields, "record_gradient_norms"));
    p.save_optimizer_state = decode_bool(field(fields, "save_optimizer_state"));
    p.save_rng_state = decode_bool(field(fields, "save_rng_state"));
    p.execution_enabled = decode_bool(field(fields, "execution_enabled"));

    // Checking every expected key individually rejects unknown fields even
    // when an expected field was removed to keep the total count unchanged.
    p.validate();
    return p;
}

} // namespace latent::nn::training
