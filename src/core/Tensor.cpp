
#include "latent/core/Tensor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace latent::core {

Tensor::Tensor(Shape shape)
    : shape_(std::move(shape)),
      values_(calculate_size(shape_), 0.0)
{
}

const Tensor::Shape& Tensor::shape() const noexcept
{
    return shape_;
}

std::size_t Tensor::rank() const noexcept
{
    return shape_.size();
}

std::size_t Tensor::size() const noexcept
{
    return values_.size();
}

bool Tensor::same_shape(const Tensor& other) const noexcept
{
    return shape_ == other.shape_;
}

double& Tensor::operator[](std::size_t index)
{
    return values_.at(index);
}

const double& Tensor::operator[](std::size_t index) const
{
    return values_.at(index);
}

double& Tensor::at(
    std::initializer_list<std::size_t> indices
)
{
    return values_.at(flatten_index(indices));
}

const double& Tensor::at(
    std::initializer_list<std::size_t> indices
) const
{
    return values_.at(flatten_index(indices));
}

double* Tensor::data() noexcept
{
    return values_.data();
}

const double* Tensor::data() const noexcept
{
    return values_.data();
}

void Tensor::fill(double value)
{
    // Non-finite values may be introduced explicitly
    // for validation tests. all_finite() detects them.
    std::fill(values_.begin(), values_.end(), value);
}

bool Tensor::all_finite() const noexcept
{
    return std::all_of(
        values_.begin(),
        values_.end(),
        [](double value) {
            return std::isfinite(value);
        }
    );
}

std::size_t Tensor::calculate_size(const Shape& shape)
{
    if (shape.empty() || shape.size() > 4) {
        throw std::invalid_argument(
            "Tensor rank must be between 1 and 4"
        );
    }

    std::size_t total = 1;

    for (const std::size_t dimension : shape) {
        if (dimension == 0) {
            throw std::invalid_argument(
                "Tensor dimensions must be positive"
            );
        }

        // Detect multiplication overflow before
        // allocating the underlying storage.
        if (total >
            std::numeric_limits<std::size_t>::max()
                / dimension) {
            throw std::overflow_error(
                "Tensor element count overflow"
            );
        }

        total *= dimension;
    }

    return total;
}

std::size_t Tensor::flatten_index(
    std::initializer_list<std::size_t> indices
) const
{
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument(
            "Tensor index rank mismatch"
        );
    }

    std::size_t offset = 0;
    auto coordinate = indices.begin();

    for (const std::size_t dimension : shape_) {
        if (*coordinate >= dimension) {
            throw std::out_of_range(
                "Tensor coordinate out of bounds"
            );
        }

        // Row-major flattening:
        // offset = offset * dimension + coordinate.
        // Safe because the shape product was validated.
        offset = offset * dimension + *coordinate;
        ++coordinate;
    }

    return offset;
}

} // namespace latent::core
