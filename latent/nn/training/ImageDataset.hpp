#pragma once

#include "latent/nn/training/ImageBatch.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace latent::nn::training {

/** @brief Immutable train, validation and test split identifiers. */
enum class ImageSplit { Train, Validation, Test };

/** @brief One RGB or grayscale image and globally unique sample ID. */
struct ImageSample {
    std::string id;
    ImageSplit split;
    latent::core::Tensor image;  // [C,H,W], finite pixels in [0,1]
};

class ImageBatchSource;

/**
 * @brief In-memory, immutable synthetic/reference image dataset.
 *
 * @details Validates global sample-ID disjointness, required nonempty
 * train/validation/test splits, uniform NCHW layout and bounded pixels.
 * Computes a deterministic, non-cryptographic content fingerprint:
 * this detects accidental mismatches, not malicious substitutions.
 *
 * Real image decoding, file hashes and a persisted signed manifest
 * belong to the later data-ingestion milestone.
 */
class ImageDataset final {
public:
    /** @brief Validate and own reference image samples. */
    explicit ImageDataset(std::vector<ImageSample> samples);

    /** @return Stable content fingerprint, prefixed with fnv1a64:. */
    [[nodiscard]] const std::string& manifest_id() const noexcept;

    /** @return Number of samples in the requested split. */
    [[nodiscard]] std::size_t split_size(ImageSplit split) const noexcept;

    /**
     * @brief Create a batch source with explicit fixed seed and size.
     * @warning The dataset must outlive the returned source.
     */
    [[nodiscard]] ImageBatchSource source(
        ImageSplit split, std::size_t batch_size,
        std::uint64_t seed) const;

private:
    friend class ImageBatchSource;
    std::vector<ImageSample> samples_;
    std::string manifest_id_;
    std::size_t train_count_ = 0;
    std::size_t validation_count_ = 0;
    std::size_t test_count_ = 0;
};

/**
 * @brief Deterministic split iterator with restartable epoch cursor.
 *
 * @details Uses a fixed SplitMix64/Fisher-Yates shuffle for training
 * and canonical sample-ID order for held-out splits. The shuffle does
 * not use std::shuffle or implementation-defined distributions.
 * A failed batch assembly does not advance the cursor.
 */
class ImageBatchSource final {
public:
    /** @return Next batch, or nullopt at end of current epoch. */
    [[nodiscard]] std::optional<ImageBatch> next();

    /** @brief Advance to the next deterministic epoch. */
    void next_epoch();

    /** @return Versioned deterministic cursor encoding. */
    [[nodiscard]] std::string checkpoint_cursor() const;

    /**
     * @brief Restore cursor only when manifest, split, size and seed match.
     * @throws std::invalid_argument Corruption or incompatible cursor.
     * @throws std::overflow_error Oversized numeric field.
     */
    void restore_cursor(const std::string& encoded);

    [[nodiscard]] std::size_t epoch() const noexcept;
    [[nodiscard]] std::size_t position() const noexcept;

private:
    friend class ImageDataset;
    ImageBatchSource(const ImageDataset& dataset, ImageSplit split,
                     std::size_t batch_size, std::uint64_t seed);
    [[nodiscard]] std::vector<std::size_t> order_for(std::size_t epoch) const;

    const ImageDataset* dataset_; // non-owning
    ImageSplit split_;
    std::size_t batch_size_;
    std::uint64_t seed_;
    std::size_t epoch_ = 0;
    std::size_t position_ = 0;
    std::vector<std::size_t> order_;
};

} // namespace latent::nn::training
