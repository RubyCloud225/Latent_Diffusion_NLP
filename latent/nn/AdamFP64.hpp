
#pragma once

#include "latent/nn/Parameter.hpp"

#include <cstddef>
#include <vector>

namespace latent::nn {

/**
 * @brief Production FP64 Adam optimiser.
 *
 * @details Updates registered Parameter tensors using
 * Adam with bias-corrected first and second moments.
 *
 * The optimiser owns its moment history but does not
 * own the registered Parameters.
 *
 * All parameter updates are staged and validated before
 * any parameter value or optimiser state is committed.
 *
 * @note Gradients are not automatically cleared by step().
 *       Call zero_grad() before the next backward pass.
 *
 * @warning Parameters must outlive this optimiser.
 * @warning Not thread-safe.
 */
class AdamFP64 final {
public:
    /**
     * @brief Construct an optimiser.
     *
     * @param parameters Non-empty list of unique,
     *        non-null, non-owning Parameter pointers.
     * @param learning_rate Positive FP64 learning rate.
     * @param beta1 First-moment decay in [0, 1).
     * @param beta2 Second-moment decay in [0, 1).
     * @param epsilon Positive denominator stabiliser.
     *
     * @throws std::invalid_argument Invalid parameters
     *         or hyperparameters.
     */
    explicit AdamFP64(
        std::vector<Parameter*> parameters,
        double learning_rate = 1e-3,
        double beta1 = 0.9,
        double beta2 = 0.999,
        double epsilon = 1e-8
    );

    /**
     * @brief Perform one Adam optimisation step.
     *
     * @details Uses accumulated Parameter gradients.
     * A failed step leaves parameter values, moment
     * history and step count unchanged.
     *
     * @throws std::invalid_argument Non-finite values,
     *         gradients or incompatible parameter shapes.
     * @throws std::overflow_error Non-finite arithmetic
     *         or exhausted step counter.
     */
    void step();

    /**
     * @brief Clear all registered parameter gradients.
     *
     * Does not change parameter values or Adam moments.
     */
    void zero_grad();

    /**
     * @brief Reset optimiser moments and step counter.
     *
     * Does not alter parameter values or gradients.
     */
    void reset();

    /** @return Number of successful optimisation steps. */
    std::size_t step_count() const noexcept;

private:
    struct State {
        explicit State(
            const latent::core::Tensor::Shape& shape
        )
            : first(shape), second(shape) {}

        latent::core::Tensor first;
        latent::core::Tensor second;
    };

    std::vector<Parameter*> parameters_;
    std::vector<State> states_;

    double learning_rate_;
    double beta1_;
    double beta2_;
    double epsilon_;

    std::size_t step_ = 0;
};

} // namespace latent::nn
