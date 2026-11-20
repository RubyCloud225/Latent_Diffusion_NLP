#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace latent::core {

/**
 * @brief Owning FP64 contiguous CPU Tensor in row-major order.
 *
 * @details Supports ranks 1 through 4 with positive dimensions; no
 * broadcasting, GPU operations, or implicit arithmetic. The final
 * dimension is contiguous. Flat and multidimensional indexing are checked.
 */
class Tensor final {
public:
    /** @brief Tensor dimensions in row-major order. */
    using Shape = std::vector<std::size_t>;

    /** @brief Allocate zero-initialised storage.
     * @param shape Positive dimensions, rank 1 through 4.
     * @throws std::invalid_argument Empty, zero, or unsupported rank.
     * @throws std::overflow_error Element count overflow.
     */
    explicit Tensor(Shape shape);

    /** @return Immutable shape of this Tensor. */
    const Shape& shape() const noexcept;
    /** @return Number of dimensions. */
    std::size_t rank() const noexcept;
    /** @return Number of FP64 elements. */
    std::size_t size() const noexcept;
    /** @return Whether all dimensions match another Tensor exactly. */
    bool same_shape(const Tensor& other) const noexcept;

    /** @brief Checked mutable flat element access.
     * @throws std::out_of_range Invalid flat index.
     */
    double& operator[](std::size_t index);
    /** @brief Checked read-only flat element access.
     * @throws std::out_of_range Invalid flat index.
     */
    const double& operator[](std::size_t index) const;

    /** @brief Checked mutable multidimensional access.
     * @throws std::invalid_argument Index rank mismatch.
     * @throws std::out_of_range Coordinate outside a dimension.
     */
    double& at(std::initializer_list<std::size_t> indices);
    /** @brief Checked read-only multidimensional access.
     * @throws std::invalid_argument Index rank mismatch.
     * @throws std::out_of_range Coordinate outside a dimension.
     */
    const double& at(std::initializer_list<std::size_t> indices) const;

    /** @return Pointer to contiguous mutable FP64 storage. */
    double* data() noexcept;
    /** @return Pointer to contiguous read-only FP64 storage. */
    const double* data() const noexcept;

    /** @brief Set every element to value (including non-finite values). */
    void fill(double value);
    /** @return True if all elements are finite. */
    bool all_finite() const noexcept;

private:
    Shape shape_;
    std::vector<double> values_;

    static std::size_t calculate_size(const Shape& shape);
    std::size_t flatten_index(
        std::initializer_list<std::size_t> indices
    ) const;
};

} // namespace latent::core
