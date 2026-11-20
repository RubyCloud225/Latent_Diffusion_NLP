
#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace latent::core {

/**
 * @brief Contiguous CPU tensor with FP64 storage.
 *
 * Dimensions use row-major ordering, with the final
 * dimension contiguous.
 *
 * Supported shapes:
 *   [D]
 *   [N, D]
 *   [C, H, W]
 *   [N, C, H, W]
 *
 * Empty shapes and zero-sized dimensions are rejected.
 * This class owns its memory and does not implement
 * broadcasting, arithmetic or GPU operations.
 */
class Tensor final {
public:
    using Shape = std::vector<std::size_t>;

    explicit Tensor(Shape shape);

    // Shape and storage information.
    const Shape& shape() const noexcept;
    std::size_t rank() const noexcept;
    std::size_t size() const noexcept;

    bool same_shape(const Tensor& other) const noexcept;

    // Checked flat indexing.
    double& operator[](std::size_t index);
    const double& operator[](std::size_t index) const;

    // Checked multidimensional indexing.
    double& at(std::initializer_list<std::size_t> indices);
    const double& at(
        std::initializer_list<std::size_t> indices
    ) const;

    // Raw contiguous storage access.
    double* data() noexcept;
    const double* data() const noexcept;

    // Numerical utilities.
    void fill(double value);
    bool all_finite() const noexcept;

private:
    Shape shape_;
    std::vector<double> values_;

    static std::size_t calculate_size(
        const Shape& shape
    );

    std::size_t flatten_index(
        std::initializer_list<std::size_t> indices
    ) const;
};

} // namespace latent::core
