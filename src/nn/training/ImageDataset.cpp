#include "latent/nn/training/ImageDataset.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace latent::nn::training {
namespace {
constexpr std::uint64_t FNV_OFFSET = UINT64_C(14695981039346656037);
constexpr std::uint64_t FNV_PRIME = UINT64_C(1099511628211);

void hash_byte(std::uint64_t& h, unsigned char value)
{
    h = (h ^ static_cast<std::uint64_t>(value)) * FNV_PRIME;
}

void hash_u64(std::uint64_t& h, std::uint64_t value)
{
    for (unsigned i = 0; i < 8; ++i) {
        hash_byte(h, static_cast<unsigned char>((value >> (8 * i)) & 255u));
    }
}

void hash_string(std::uint64_t& h, const std::string& value)
{
    hash_u64(h, static_cast<std::uint64_t>(value.size()));
    for (unsigned char c : value) hash_byte(h, c);
}

std::uint64_t splitmix64(std::uint64_t& state)
{
    state += UINT64_C(0x9e3779b97f4a7c15);
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

// Uniform integer in [0,bound) with no modulo bias.
std::uint64_t bounded(std::uint64_t& state, std::uint64_t bound)
{
    const std::uint64_t threshold = (std::uint64_t(0) - bound) % bound;
    for (;;) {
        const auto value = splitmix64(state);
        if (value >= threshold) return value % bound;
    }
}

const char* split_name(ImageSplit split)
{
    switch (split) {
        case ImageSplit::Train: return "train";
        case ImageSplit::Validation: return "validation";
        case ImageSplit::Test: return "test";
    }
    throw std::invalid_argument("Unsupported image split");
}

std::size_t read_size(const std::string& token)
{
    if (token.empty() ||
        !std::all_of(token.begin(), token.end(),
                     [](char c) { return c >= '0' && c <= '9'; })) {
        throw std::invalid_argument("Invalid image cursor integer");
    }
    std::size_t used = 0;
    std::uint64_t value = 0;
    try { value = std::stoull(token, &used, 10); }
    catch (const std::out_of_range&) {
        throw std::overflow_error("Image cursor integer overflow");
    }
    if (used != token.size() ||
        value > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("Image cursor integer out of range");
    }
    return static_cast<std::size_t>(value);
}

std::uint64_t read_seed(const std::string& token)
{
    if (token.empty() ||
        !std::all_of(token.begin(), token.end(),
                     [](char c) { return c >= '0' && c <= '9'; })) {
        throw std::invalid_argument("Invalid image cursor seed");
    }
    std::size_t used = 0;
    std::uint64_t value = 0;
    try { value = std::stoull(token, &used, 10); }
    catch (const std::out_of_range&) {
        throw std::overflow_error("Image cursor seed overflow");
    }
    if (used != token.size()) throw std::invalid_argument("Invalid cursor seed");
    return value;
}
} // namespace

ImageDataset::ImageDataset(std::vector<ImageSample> samples)
    : samples_(std::move(samples))
{
    if (samples_.empty()) throw std::invalid_argument("Empty image dataset");
    std::sort(samples_.begin(), samples_.end(),
        [](const ImageSample& a, const ImageSample& b) {
            return a.id < b.id;
        });

    std::unordered_set<std::string> seen;
    std::size_t channels = 0, height = 0, width = 0;
    std::uint64_t fingerprint = FNV_OFFSET;
    hash_string(fingerprint, "image_dataset_v1");

    for (const auto& sample : samples_) {
        if (sample.id.empty() ||
            !seen.insert(sample.id).second) {
            throw std::invalid_argument("Empty or repeated image sample ID across splits");
        }
        for (unsigned char c : sample.id) {
            if (c < 32 || c == 127) {
                throw std::invalid_argument("Invalid control character in sample ID");
            }
        }

        const auto& image = sample.image;
        if (image.rank() != 3 || !image.all_finite()) {
            throw std::invalid_argument("Image sample requires finite CHW Tensor");
        }
        const auto& shape = image.shape();
        if (shape[0] != 1 && shape[0] != 3) {
            throw std::invalid_argument("Reference images require 1 or 3 channels");
        }
        if (channels == 0) {
            channels = shape[0];
            height = shape[1];
            width = shape[2];
        }
        if (shape[0] != channels || shape[1] != height ||
            shape[2] != width) {
            throw std::invalid_argument("Inconsistent reference image shape");
        }

        switch (sample.split) {
            case ImageSplit::Train: ++train_count_; break;
            case ImageSplit::Validation: ++validation_count_; break;
            case ImageSplit::Test: ++test_count_; break;
            default: throw std::invalid_argument("Unsupported image split");
        }

        hash_string(fingerprint, sample.id);
        hash_string(fingerprint, split_name(sample.split));
        for (std::size_t d : shape) hash_u64(fingerprint, d);
        for (std::size_t i = 0; i < image.size(); ++i) {
            const double pixel = image[i];
            if (pixel < 0.0 || pixel > 1.0) {
                throw std::invalid_argument("Image pixel outside [0,1]");
            }
            std::uint64_t bits = 0;
            static_assert(sizeof(bits) == sizeof(pixel), "Requires FP64 binary64");
            std::memcpy(&bits, &pixel, sizeof(bits));
            hash_u64(fingerprint, bits);
        }
    }

    if (train_count_ == 0 || validation_count_ == 0 || test_count_ == 0) {
        throw std::invalid_argument("All three image splits must have samples");
    }

    std::ostringstream output;
    output << "fnv1a64:" << std::hex << std::setfill('0')
           << std::setw(16) << fingerprint;
    manifest_id_ = output.str();
}

const std::string& ImageDataset::manifest_id() const noexcept
{
    return manifest_id_;
}

const latent::core::Tensor::Shape&
ImageDataset::image_shape() const noexcept
{
    return samples_.front().image.shape();
}

std::size_t ImageDataset::split_size(ImageSplit split) const noexcept
{
    switch (split) {
        case ImageSplit::Train: return train_count_;
        case ImageSplit::Validation: return validation_count_;
        case ImageSplit::Test: return test_count_;
    }
    return 0;
}

ImageBatchSource ImageDataset::source(
    ImageSplit split, std::size_t batch_size, std::uint64_t seed) const
{
    return ImageBatchSource(*this, split, batch_size, seed);
}

ImageBatchSource::ImageBatchSource(
    const ImageDataset& dataset, ImageSplit split,
    std::size_t batch_size, std::uint64_t seed)
    : dataset_(&dataset), split_(split), batch_size_(batch_size), seed_(seed)
{
    if (batch_size == 0) {
        throw std::invalid_argument("Image batch size must be positive");
    }
    (void)split_name(split_);
    order_ = order_for(epoch_);
}

std::vector<std::size_t> ImageBatchSource::order_for(
    std::size_t epoch) const
{
    std::vector<std::size_t> order;
    for (std::size_t i = 0; i < dataset_->samples_.size(); ++i) {
        if (dataset_->samples_[i].split == split_) order.push_back(i);
    }

    if (split_ == ImageSplit::Train) {
        std::uint64_t state = seed_ ^ UINT64_C(0x6a09e667f3bcc909);
        state ^= static_cast<std::uint64_t>(epoch);
        for (std::size_t i = order.size(); i > 1; --i) {
            const std::size_t j = static_cast<std::size_t>(
                bounded(state, static_cast<std::uint64_t>(i))
            );
            std::swap(order[i - 1], order[j]);
        }
    }
    return order;
}

std::optional<ImageBatch> ImageBatchSource::next()
{
    if (position_ == order_.size()) return std::nullopt;

    const std::size_t remaining = order_.size() - position_;
    const std::size_t count = std::min(batch_size_, remaining);
    const auto& first = dataset_->samples_[order_[position_]].image;
    const auto& shape = first.shape();
    latent::core::Tensor pixels({count, shape[0], shape[1], shape[2]});
    std::vector<std::string> ids;
    ids.reserve(count);

    const std::size_t per_image = first.size();
    for (std::size_t sample = 0; sample < count; ++sample) {
        const auto& item = dataset_->samples_[order_[position_ + sample]];
        ids.push_back(item.id);
        for (std::size_t p = 0; p < per_image; ++p) {
            pixels[sample * per_image + p] = item.image[p];
        }
    }

    ImageBatch result(std::move(pixels), std::move(ids));
    position_ += count; // Commit cursor only after validated assembly.
    return result;
}

void ImageBatchSource::next_epoch()
{
    if (epoch_ == std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("Image epoch counter exhausted");
    }
    const auto next = order_for(epoch_ + 1);
    order_ = next;
    ++epoch_;
    position_ = 0;
}

std::string ImageBatchSource::checkpoint_cursor() const
{
    return std::string("image_cursor_v1|") + dataset_->manifest_id() + "|" +
        split_name(split_) + "|" + std::to_string(batch_size_) + "|" +
        std::to_string(seed_) + "|" + std::to_string(epoch_) + "|" +
        std::to_string(position_);
}

void ImageBatchSource::restore_cursor(const std::string& encoded)
{
    std::vector<std::string> parts;
    std::istringstream input(encoded);
    std::string part;
    while (std::getline(input, part, '|')) parts.push_back(part);
    if (parts.size() != 7 || parts[0] != "image_cursor_v1" ||
        parts[1] != dataset_->manifest_id() ||
        parts[2] != split_name(split_) ||
        read_size(parts[3]) != batch_size_ ||
        read_seed(parts[4]) != seed_) {
        throw std::invalid_argument("Image cursor incompatible with source");
    }

    const std::size_t proposed_epoch = read_size(parts[5]);
    const std::size_t proposed_position = read_size(parts[6]);
    auto proposed_order = order_for(proposed_epoch);
    if (proposed_position > proposed_order.size()) {
        throw std::invalid_argument("Image cursor beyond split end");
    }
    order_ = std::move(proposed_order);
    epoch_ = proposed_epoch;
    position_ = proposed_position;
}

std::size_t ImageBatchSource::epoch() const noexcept { return epoch_; }
std::size_t ImageBatchSource::position() const noexcept { return position_; }

} // namespace latent::nn::training
