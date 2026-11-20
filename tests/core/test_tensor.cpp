
#include "latent/core/Tensor.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using latent::core::Tensor;

namespace {

void require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template<typename Exception, typename Function>
void expect_throw(Function function)
{
    bool caught = false;

    try {
        function();
    } catch (const Exception&) {
        caught = true;
    }

    require(caught, "Expected exception was not thrown");
}

void test_constructor()
{
    Tensor tensor({2, 3, 4});

    require(tensor.rank() == 3, "Incorrect rank");
    require(tensor.size() == 24, "Incorrect size");
    require(
        tensor.shape() == Tensor::Shape({2, 3, 4}),
        "Incorrect shape"
    );

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        require(tensor[i] == 0.0, "Not zero initialised");
    }
}

void test_supported_ranks()
{
    require(Tensor({5}).size() == 5, "Rank 1 failed");
    require(Tensor({2, 3}).size() == 6, "Rank 2 failed");
    require(Tensor({2, 3, 4}).size() == 24, "Rank 3 failed");
    require(
        Tensor({2, 3, 4, 5}).size() == 120,
        "Rank 4 failed"
    );
}

void test_flat_indexing()
{
    Tensor tensor({2, 3});

    tensor[0] = 1.0;
    tensor[5] = 6.0;

    const Tensor& readonly = tensor;

    require(readonly[0] == 1.0, "First value incorrect");
    require(readonly[5] == 6.0, "Last value incorrect");

    expect_throw<std::out_of_range>([&] {
        (void)tensor[6];
    });
}

void test_multidimensional_indexing()
{
    Tensor tensor({2, 3, 4});

    tensor.at({1, 2, 3}) = 42.0;

    require(tensor[23] == 42.0, "Flattening failed");

    const Tensor& readonly = tensor;
    require(
        readonly.at({1, 2, 3}) == 42.0,
        "Const indexing failed"
    );

    expect_throw<std::out_of_range>([&] {
        (void)tensor.at({2, 0, 0});
    });

    expect_throw<std::invalid_argument>([&] {
        (void)tensor.at({1, 2});
    });
}

void test_contiguous_storage()
{
    Tensor tensor({2, 3});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<double>(i);
    }

    const double* pointer = tensor.data();

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        require(
            *(pointer + i) == static_cast<double>(i),
            "Contiguous storage failed"
        );
    }
}

void test_fill()
{
    Tensor tensor({3, 4});
    tensor.fill(2.5);

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        require(tensor[i] == 2.5, "Fill failed");
    }
}

void test_shape_comparison()
{
    Tensor a({2, 3});
    Tensor b({2, 3});
    Tensor c({3, 2});

    require(a.same_shape(b), "Equal shapes rejected");
    require(!a.same_shape(c), "Different shapes accepted");
}

void test_finite_validation()
{
    Tensor tensor({3});
    tensor.fill(1.0);

    require(tensor.all_finite(), "Finite values rejected");

    tensor[0] = std::numeric_limits<double>::infinity();

    require(!tensor.all_finite(), "Infinity not detected");

    tensor[0] = std::numeric_limits<double>::quiet_NaN();

    require(!tensor.all_finite(), "NaN not detected");
}

void test_invalid_dimensions()
{
    expect_throw<std::invalid_argument>([] {
        Tensor tensor({});
    });

    expect_throw<std::invalid_argument>([] {
        Tensor tensor({2, 0, 3});
    });

    expect_throw<std::invalid_argument>([] {
        Tensor tensor({1, 1, 1, 1, 1});
    });

    expect_throw<std::overflow_error>([] {
        Tensor tensor({
            std::numeric_limits<std::size_t>::max(), 2
        });
    });
}

void test_copy_isolation()
{
    Tensor original({2});
    original.fill(1.0);

    Tensor copy = original;
    copy[0] = 99.0;

    require(original[0] == 1.0, "Copy aliases storage");
    require(copy[0] == 99.0, "Copy update failed");
}

} // namespace latent::core

int main()
{
    try {
        test_constructor();
        test_supported_ranks();
        test_flat_indexing();
        test_multidimensional_indexing();
        test_contiguous_storage();
        test_fill();
        test_shape_comparison();
        test_finite_validation();
        test_invalid_dimensions();
        test_copy_isolation();

        std::cout << "[PASS] P0.4 Tensor tests\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "[FAIL] P0.4 Tensor: "
            << error.what() << '\n';
        return 1;
    }
}
