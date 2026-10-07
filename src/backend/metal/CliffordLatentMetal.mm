#include "latent/backend/metal/CliffordProjectionMetal.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CLIFFORD_METALLIB_PATH
#error "CLIFFORD_METALLIB_PATH must be defined by CMake."
#endif

namespace latent::backend::metal {

namespace {

id<MTLLibrary> load_library(
    id<MTLDevice> device
)
{
    NSString* path =
        [NSString stringWithUTF8String:
            CLIFFORD_METALLIB_PATH];

    NSError* error = nil;

    NSURL* url =
        [NSURL fileURLWithPath:path];

    id<MTLLibrary> library =
        [device newLibraryWithURL:url
                            error:&error];

    if (library == nil) {
        std::string message =
            "Failed to load Metal library";

        if (error != nil) {
            message += ": ";
            message +=
                [[error localizedDescription]
                    UTF8String];
        }

        throw std::runtime_error(
            message
        );
    }

    return library;
}

} // namespace


latent::experimental::CliffordMultivector
CliffordProjectionMetal::forward(
    const std::vector<float>& input,
    const Matrix& weights
)
{
    if (
        input.size() !=
        Projection::SourceDimensions
    ) {
        throw std::invalid_argument(
            "CliffordProjectionMetal::forward "
            "expected a 64-dimensional input."
        );
    }

    id<MTLDevice> device =
        MTLCreateSystemDefaultDevice();

    if (device == nil) {
        throw std::runtime_error(
            "Metal device is unavailable."
        );
    }

    id<MTLCommandQueue> queue =
        [device newCommandQueue];

    if (queue == nil) {
        throw std::runtime_error(
            "Failed to create Metal command queue."
        );
    }

    id<MTLLibrary> library =
        load_library(device);

    id<MTLFunction> function =
        [library
            newFunctionWithName:
                @"clifford_projection_forward"];

    if (function == nil) {
        throw std::runtime_error(
            "Metal function "
            "'clifford_projection_forward' "
            "was not found."
        );
    }

    NSError* pipelineError = nil;

    id<MTLComputePipelineState> pipeline =
        [device
            newComputePipelineStateWithFunction:
                function
            error:
                &pipelineError];

    if (pipeline == nil) {
        std::string message =
            "Failed to create Metal compute pipeline";

        if (pipelineError != nil) {
            message += ": ";
            message +=
                [[pipelineError localizedDescription]
                    UTF8String];
        }

        throw std::runtime_error(
            message
        );
    }

    constexpr std::size_t inputBytes =
        Projection::SourceDimensions *
        sizeof(float);

    constexpr std::size_t weightBytes =
        Projection::BladeCount *
        Projection::SourceDimensions *
        sizeof(float);

    constexpr std::size_t outputFP32Bytes =
        Projection::BladeCount *
        sizeof(float);

    constexpr std::size_t outputFP16Bytes =
        Projection::BladeCount *
        sizeof(std::uint16_t);

    id<MTLBuffer> inputBuffer =
        [device
            newBufferWithBytes:
                input.data()
            length:
                inputBytes
            options:
                MTLResourceStorageModeShared];

    id<MTLBuffer> weightBuffer =
        [device
            newBufferWithBytes:
                weights.data()
            length:
                weightBytes
            options:
                MTLResourceStorageModeShared];

    id<MTLBuffer> outputFP32Buffer =
        [device
            newBufferWithLength:
                outputFP32Bytes
            options:
                MTLResourceStorageModeShared];

    id<MTLBuffer> outputFP16Buffer =
        [device
            newBufferWithLength:
                outputFP16Bytes
            options:
                MTLResourceStorageModeShared];

    if (
        inputBuffer == nil ||
        weightBuffer == nil ||
        outputFP32Buffer == nil ||
        outputFP16Buffer == nil
    ) {
        throw std::runtime_error(
            "Failed to allocate Metal buffers."
        );
    }

    id<MTLCommandBuffer> commandBuffer =
        [queue commandBuffer];

    if (commandBuffer == nil) {
        throw std::runtime_error(
            "Failed to create Metal command buffer."
        );
    }

    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];

    if (encoder == nil) {
        throw std::runtime_error(
            "Failed to create Metal compute encoder."
        );
    }

    [encoder
        setComputePipelineState:
            pipeline];

    [encoder
        setBuffer:
            inputBuffer
        offset:
            0
        atIndex:
            0];

    [encoder
        setBuffer:
            weightBuffer
        offset:
            0
        atIndex:
            1];

    [encoder
        setBuffer:
            outputFP32Buffer
        offset:
            0
        atIndex:
            2];

    [encoder
        setBuffer:
            outputFP16Buffer
        offset:
            0
        atIndex:
            3];

    const MTLSize grid =
        MTLSizeMake(
            Projection::BladeCount,
            1,
            1
        );

    const NSUInteger width =
        pipeline.threadExecutionWidth;

    const NSUInteger groupWidth =
        std::min<NSUInteger>(
            width,
            Projection::BladeCount
        );

    const MTLSize threadsPerGroup =
        MTLSizeMake(
            groupWidth,
            1,
            1
        );

    [encoder
        dispatchThreads:
            grid
        threadsPerThreadgroup:
            threadsPerGroup];

    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    if (
        commandBuffer.status ==
        MTLCommandBufferStatusError
    ) {
        std::string message =
            "Metal command buffer failed";

        if (commandBuffer.error != nil) {
            message += ": ";
            message +=
                [[commandBuffer.error
                    localizedDescription]
                    UTF8String];
        }

        throw std::runtime_error(
            message
        );
    }

    latent::experimental::
        CliffordMultivector result{};

    std::memcpy(
        result.fp32.data(),
        [outputFP32Buffer contents],
        outputFP32Bytes
    );

    std::memcpy(
        result.fp16.data(),
        [outputFP16Buffer contents],
        outputFP16Bytes
    );

    return result;
}
CliffordProjectionMetal::BackwardResult
CliffordProjectionMetal::backward(
    const std::vector<float>& input,
    const std::array<float, Projection::BladeCount>& grad_output,
    const Matrix& weights
)
{
    if (
        input.size() !=
        Projection::SourceDimensions
    ) {
        throw std::invalid_argument(
            "CliffordProjectionMetal::backward "
            "expected a 64-dimensional input."
        );
    }

    id<MTLDevice> device =
        MTLCreateSystemDefaultDevice();

    if (device == nil) {
        throw std::runtime_error(
            "Metal device is unavailable."
        );
    }

    id<MTLCommandQueue> queue =
        [device newCommandQueue];

    if (queue == nil) {
        throw std::runtime_error(
            "Failed to create Metal command queue."
        );
    }

    id<MTLLibrary> library =
        load_library(device);

    id<MTLFunction> weightFunction =
        [library
            newFunctionWithName:
                @"clifford_projection_backward_weights"];

    id<MTLFunction> inputFunction =
        [library
            newFunctionWithName:
                @"clifford_projection_backward_input"];

    if (
        weightFunction == nil ||
        inputFunction == nil
    ) {
        throw std::runtime_error(
            "Failed to load Clifford backward Metal functions."
        );
    }

    NSError* error = nil;

    id<MTLComputePipelineState> weightPipeline =
        [device
            newComputePipelineStateWithFunction:
                weightFunction
            error:
                &error];

    if (weightPipeline == nil) {
        throw std::runtime_error(
            "Failed to create Metal weight-gradient pipeline."
        );
    }

    id<MTLComputePipelineState> inputPipeline =
        [device
            newComputePipelineStateWithFunction:
                inputFunction
            error:
                &error];

    if (inputPipeline == nil) {
        throw std::runtime_error(
            "Failed to create Metal input-gradient pipeline."
        );
    }

    constexpr std::size_t inputBytes =
        Projection::SourceDimensions *
        sizeof(float);

    constexpr std::size_t weightBytes =
        Projection::BladeCount *
        Projection::SourceDimensions *
        sizeof(float);

    constexpr std::size_t gradOutputBytes =
        Projection::BladeCount *
        sizeof(float);

    constexpr std::size_t gradInputBytes =
        Projection::SourceDimensions *
        sizeof(float);

    constexpr std::size_t gradWeightBytes =
        Projection::BladeCount *
        Projection::SourceDimensions *
        sizeof(float);

    id<MTLBuffer> inputBuffer =
        [device
            newBufferWithBytes:
                input.data()
            length:
                inputBytes
            options:
                MTLResourceStorageModeShared];

    id<MTLBuffer> weightBuffer =
        [device
            newBufferWithBytes:
                weights.data()
            length:
                weightBytes
            options:
                MTLResourceStorageModeShared];

    id<MTLBuffer> gradOutputBuffer =
        [device
            newBufferWithBytes:
                grad_output.data()
            length:
                gradOutputBytes
            options:
                MTLResourceStorageModeShared];

    id<MTLBuffer> gradWeightBuffer =
        [device
            newBufferWithLength:
                gradWeightBytes
            options:
                MTLResourceStorageModeShared];

    id<MTLBuffer> gradInputBuffer =
        [device
            newBufferWithLength:
                gradInputBytes
            options:
                MTLResourceStorageModeShared];

    if (
        inputBuffer == nil ||
        weightBuffer == nil ||
        gradOutputBuffer == nil ||
        gradWeightBuffer == nil ||
        gradInputBuffer == nil
    ) {
        throw std::runtime_error(
            "Failed to allocate Metal backward buffers."
        );
    }

    id<MTLCommandBuffer> commandBuffer =
        [queue commandBuffer];

    if (commandBuffer == nil) {
        throw std::runtime_error(
            "Failed to create Metal command buffer."
        );
    }

    // -----------------------------------------
    // Weight gradients
    // -----------------------------------------

    id<MTLComputeCommandEncoder> weightEncoder =
        [commandBuffer computeCommandEncoder];

    [weightEncoder
        setComputePipelineState:
            weightPipeline];

    [weightEncoder
        setBuffer:
            inputBuffer
        offset:
            0
        atIndex:
            0];

    [weightEncoder
        setBuffer:
            gradOutputBuffer
        offset:
            0
        atIndex:
            1];

    [weightEncoder
        setBuffer:
            gradWeightBuffer
        offset:
            0
        atIndex:
            2];

    const MTLSize weightGrid =
        MTLSizeMake(
            Projection::BladeCount *
            Projection::SourceDimensions,
            1,
            1
        );

    const NSUInteger weightThreadWidth =
        weightPipeline.threadExecutionWidth;

    const MTLSize weightGroup =
        MTLSizeMake(
            std::min<NSUInteger>(
                weightThreadWidth,
                Projection::BladeCount *
                Projection::SourceDimensions
            ),
            1,
            1
        );

    [weightEncoder
        dispatchThreads:
            weightGrid
        threadsPerThreadgroup:
            weightGroup];

    [weightEncoder endEncoding];

    // -----------------------------------------
    // Input gradients
    // -----------------------------------------

    id<MTLComputeCommandEncoder> inputEncoder =
        [commandBuffer computeCommandEncoder];

    [inputEncoder
        setComputePipelineState:
            inputPipeline];

    [inputEncoder
        setBuffer:
            weightBuffer
        offset:
            0
        atIndex:
            0];

    [inputEncoder
        setBuffer:
            gradOutputBuffer
        offset:
            0
        atIndex:
            1];

    [inputEncoder
        setBuffer:
            gradInputBuffer
        offset:
            0
        atIndex:
            2];

    const MTLSize inputGrid =
        MTLSizeMake(
            Projection::SourceDimensions,
            1,
            1
        );

    const NSUInteger inputThreadWidth =
        inputPipeline.threadExecutionWidth;

    const MTLSize inputGroup =
        MTLSizeMake(
            std::min<NSUInteger>(
                inputThreadWidth,
                Projection::SourceDimensions
            ),
            1,
            1
        );

    [inputEncoder
        dispatchThreads:
            inputGrid
        threadsPerThreadgroup:
            inputGroup];

    [inputEncoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    if (
        commandBuffer.status ==
        MTLCommandBufferStatusError
    ) {
        throw std::runtime_error(
            "Metal Clifford backward command failed."
        );
    }

    BackwardResult result;

    result.grad_input.resize(
        Projection::SourceDimensions
    );

    std::memcpy(
        result.grad_input.data(),
        [gradInputBuffer contents],
        gradInputBytes
    );

    std::memcpy(
        result.grad_weights.data(),
        [gradWeightBuffer contents],
        gradWeightBytes
    );

    return result;
}

} // namespace latent::backend::metal
