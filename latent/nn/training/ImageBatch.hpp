#pragma once

#include "latent/core/Tensor.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace latent::nn::training {

/**
 * @brief One validated FP64 NCHW image microbatch.
 *
 * @details The reference dataset uses finite [0,1] pixels and an
 * explicit image layout. Sample identifiers preserve provenance.
 * Channels, height and width are constant within a source.
 */
struct ImageBatch final {
    latent::core::Tensor images;
    std::vector<std::string> sample_ids;

    /** @brief Construct and validate a microbatch. */
    ImageBatch(latent::core::Tensor pixels,
               std::vector<std::string> ids)
        : images(std::move(pixels)), sample_ids(std::move(ids))
    {
        validate();
    }

    /** @brief Validate rank, IDs, finiteness and pixel range. */
    void validate() const
    {
        if (images.rank() != 4 || images.shape()[0] != sample_ids.size()) {
            throw std::invalid_argument("ImageBatch requires matching NCHW images and IDs");
        }
        if (!images.all_finite()) {
            throw std::invalid_argument("ImageBatch pixels must be finite");
        }
        for (const std::string& id : sample_ids) {
            if (id.empty()) {
                throw std::invalid_argument("ImageBatch sample ID is empty");
            }
        }
        for (std::size_t i = 0; i < images.size(); ++i) {
            if (images[i] < 0.0 || images[i] > 1.0) {
                throw std::invalid_argument("ImageBatch pixels must be in [0,1]");
            }
        }
    }

    /** @return Number of images in the microbatch. */
    [[nodiscard]] std::size_t sample_count() const noexcept
    {
        return images.shape()[0];
    }
};

} // namespace latent::nn::training
