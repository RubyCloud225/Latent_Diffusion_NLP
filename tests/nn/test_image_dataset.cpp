#include "latent/nn/training/ImageDataset.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using latent::core::Tensor;
using latent::nn::training::ImageBatch;
using latent::nn::training::ImageBatchSource;
using latent::nn::training::ImageDataset;
using latent::nn::training::ImageSample;
using latent::nn::training::ImageSplit;

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

template<typename Exception, typename F>
void rejects(F operation)
{
    bool caught = false;
    try { operation(); }
    catch (const Exception&) { caught = true; }
    require(caught, "Expected exception not raised");
}

ImageSample sample(const std::string& id, ImageSplit split, double value)
{
    Tensor image({1, 2, 2});
    image.fill(value);
    return {id, split, std::move(image)};
}

ImageDataset fixture()
{
    return ImageDataset({
        sample("train-3", ImageSplit::Train, 0.3),
        sample("validation-1", ImageSplit::Validation, 0.7),
        sample("train-1", ImageSplit::Train, 0.1),
        sample("test-1", ImageSplit::Test, 0.9),
        sample("train-4", ImageSplit::Train, 0.4),
        sample("train-2", ImageSplit::Train, 0.2),
        sample("validation-2", ImageSplit::Validation, 0.8),
        sample("train-5", ImageSplit::Train, 0.5)
    });
}

std::vector<std::string> collect(ImageBatchSource& source)
{
    std::vector<std::string> ids;
    while (auto batch = source.next()) {
        batch->validate();
        ids.insert(ids.end(),
                   batch->sample_ids.begin(), batch->sample_ids.end());
    }
    return ids;
}

void test_nchw_and_pixel_mapping()
{
    auto dataset = fixture();
    auto source = dataset.source(ImageSplit::Train, 2, 42);
    const auto batch = source.next();
    require(batch.has_value(), "First batch missing");
    require(batch->images.shape() == Tensor::Shape({2, 1, 2, 2}),
            "Wrong NCHW shape");
    require(batch->sample_ids.size() == 2, "Missing sample IDs");

    for (std::size_t n = 0; n < batch->sample_ids.size(); ++n) {
        const std::string& id = batch->sample_ids[n];
        const double expected =
            static_cast<double>(id.back() - '0') / 10.0;
        for (std::size_t i = 0; i < 4; ++i) {
            require(batch->images[n * 4 + i] == expected,
                    "Image content mapped to wrong sample ID");
        }
    }
    require(dataset.split_size(ImageSplit::Train) == 5,
            "Wrong train count");
    require(dataset.split_size(ImageSplit::Validation) == 2,
            "Wrong validation count");
    require(dataset.split_size(ImageSplit::Test) == 1,
            "Wrong test count");
}

void test_deterministic_replay_and_cursor()
{
    auto dataset = fixture();
    auto a = dataset.source(ImageSplit::Train, 2, 123);
    auto b = dataset.source(ImageSplit::Train, 2, 123);

    const auto first = a.next();
    require(first.has_value(), "First batch missing");
    const auto checkpoint = a.checkpoint_cursor();

    auto restored = dataset.source(ImageSplit::Train, 2, 123);
    restored.restore_cursor(checkpoint);

    const auto remainder = collect(a);
    require(remainder == collect(restored),
            "Restored cursor changed remaining image order");

    const auto original_epoch = b.checkpoint_cursor();
    require(original_epoch != checkpoint,
            "Cursor did not advance");

    auto c = dataset.source(ImageSplit::Train, 2, 123);
    require(collect(b) == collect(c),
            "Seeded epoch-zero order is not reproducible");

    a.next_epoch();
    restored.next_epoch();
    require(collect(a) == collect(restored),
            "Seeded epoch-one order is not reproducible");
    require(a.epoch() == 1 && a.position() == 5,
            "Wrong post-epoch cursor");
}

void test_partial_batch_and_disjoint_splits()
{
    auto dataset = fixture();
    auto train = dataset.source(ImageSplit::Train, 2, 0);
    std::set<std::string> train_ids;
    std::size_t batches = 0;
    while (auto batch = train.next()) {
        ++batches;
        train_ids.insert(batch->sample_ids.begin(), batch->sample_ids.end());
        if (batches == 3) {
            require(batch->sample_count() == 1,
                    "Final partial batch must be one image");
        }
    }
    require(batches == 3 && train_ids.size() == 5,
            "Missing or repeated training images");
    require(!train.next().has_value(), "End of epoch did not stop");

    auto validation = dataset.source(ImageSplit::Validation, 8, 0);
    auto test = dataset.source(ImageSplit::Test, 8, 0);
    const auto validation_ids = collect(validation);
    const auto test_ids = collect(test);
    require(validation_ids == std::vector<std::string>({
                "validation-1", "validation-2"}),
            "Held-out validation order changed");
    require(test_ids == std::vector<std::string>({"test-1"}),
            "Test split not isolated");
    for (const auto& id : validation_ids)
        require(train_ids.count(id) == 0, "Train/validation leakage");
    for (const auto& id : test_ids)
        require(train_ids.count(id) == 0, "Train/test leakage");
}

void test_manifest_content_identity()
{
    auto a = fixture();
    auto b = fixture();
    require(a.manifest_id() == b.manifest_id(),
            "Same samples have different fingerprints");

    auto changed = ImageDataset({
        sample("train-1", ImageSplit::Train, 0.12),
        sample("validation-1", ImageSplit::Validation, 0.7),
        sample("test-1", ImageSplit::Test, 0.9)
    });
    require(a.manifest_id() != changed.manifest_id(),
            "Changed dataset has same fingerprint");
}

void test_cursor_validation_atomic()
{
    auto dataset = fixture();
    auto source = dataset.source(ImageSplit::Train, 2, 42);
    const auto initial = source.checkpoint_cursor();

    rejects<std::invalid_argument>([&] {
        source.restore_cursor("image_cursor_v1|missing|train|2|42|0|0");
    });
    rejects<std::invalid_argument>([&] {
        source.restore_cursor(initial + "|unexpected");
    });
    rejects<std::invalid_argument>([&] {
        source.restore_cursor(
            "image_cursor_v1|" + dataset.manifest_id() +
            "|train|2|42|0|999");
    });
    auto another_seed = dataset.source(ImageSplit::Train, 2, 43);
    rejects<std::invalid_argument>([&] {
        another_seed.restore_cursor(initial);
    });
    require(source.checkpoint_cursor() == initial,
            "Invalid restore modified cursor");
}

void test_dataset_rejection()
{
    rejects<std::invalid_argument>([] {
        ImageDataset bad({
            sample("same", ImageSplit::Train, 0.1),
            sample("same", ImageSplit::Validation, 0.2),
            sample("test", ImageSplit::Test, 0.3)
        });
    });
    rejects<std::invalid_argument>([] {
        ImageDataset bad({
            sample("train", ImageSplit::Train, 0.1),
            sample("test", ImageSplit::Test, 0.3)
        });
    });
    rejects<std::invalid_argument>([] {
        auto invalid = sample("train", ImageSplit::Train, 0.1);
        invalid.image[0] = std::numeric_limits<double>::quiet_NaN();
        ImageDataset bad({
            std::move(invalid),
            sample("validation", ImageSplit::Validation, 0.2),
            sample("test", ImageSplit::Test, 0.3)
        });
    });
    rejects<std::invalid_argument>([] {
        ImageDataset bad({
            sample("train", ImageSplit::Train, 1.1),
            sample("validation", ImageSplit::Validation, 0.2),
            sample("test", ImageSplit::Test, 0.3)
        });
    });
    rejects<std::invalid_argument>([] {
        ImageDataset bad({
            sample("train", ImageSplit::Train, 0.1),
            ImageSample{"validation", ImageSplit::Validation, Tensor({3, 2, 2})},
            sample("test", ImageSplit::Test, 0.3)
        });
    });
    rejects<std::invalid_argument>([] {
        auto dataset = fixture();
        (void)dataset.source(ImageSplit::Train, 0, 1);
    });
    rejects<std::invalid_argument>([] {
        ImageBatch invalid(Tensor({1, 1, 2}), {"x"});
    });
}

} // namespace

int main()
{
    try {
        test_nchw_and_pixel_mapping();
        test_deterministic_replay_and_cursor();
        test_partial_batch_and_disjoint_splits();
        test_manifest_content_identity();
        test_cursor_validation_atomic();
        test_dataset_rejection();
        std::cout << "[PASS] NN.1.07E.01 deterministic image dataset\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] NN.1.07E.01: " << error.what() << '\n';
        return 1;
    }
}
