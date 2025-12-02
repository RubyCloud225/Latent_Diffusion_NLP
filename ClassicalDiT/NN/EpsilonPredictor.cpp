#include "EpsilonPredictor.hpp"
#include <vector>


EpsilonPredictor::EpsilonPredictor(int input_channels, int output_size) {
    // Constructor left empty to avoid persistent nn_ member causing thread safety issues
}

std::vector<int> EpsilonPredictor::predictEpsilon(const std::vector<double>& x_t, int t) {
    // Create a new NeuralNetwork instance per call to ensure thread safety
    NeuralNetwork nn;
    nn.addConvolutionalLayer(1, 64, 1, 1); // conv layer, input_channels=1, kernel_size=1
    nn.addReluLayer(); // Activation

    // Calculate output size after first conv layer
    int conv1_output_size = x_t.size(); // input_size - kernel_size + 1 (stride=1, kernel=1)
    int pool1_input_height = conv1_output_size;
    int pool1_input_width = conv1_output_size;

    nn.addPoolingLayer(pool1_input_height, pool1_input_width, 2, 2, 0); // pooling layer using max pooling

    nn.addConvolutionalLayer(64, 128, 1, 1); // second conv layer, kernel_size=1
    nn.addReluLayer(); // activation

    // Calculate output size after second conv layer
    int conv2_output_size = pool1_input_height / 2; // after pooling and conv2 (kernel=1)
    int pool2_input_height = conv2_output_size;
    int pool2_input_width = conv2_output_size;

    nn.addPoolingLayer(pool2_input_height, pool2_input_width, 2, 2, 0); // pooling layer
    nn.addFlatten(); // Flatten the output

    // Calculate flatten size after second pooling layer
    int flatten_size = pool2_input_height * pool2_input_width * 128;

    nn.addFullyConnectedLayer(flatten_size, x_t.size()); // output layer

    // Reshape x_t into a 2D vector with appropriate height and width
    int length = x_t.size();
    int height = static_cast<int>(std::sqrt(length));
    int width = height;
    if (height * width < length) {
        width += 1;
    }
    std::vector<std::vector<double>> input(height, std::vector<double>(width, 0.0));

    // Debug print input dimensions
    std::cout << "EpsilonPredictor::predictEpsilon reshaped input dimensions: " << height << " x " << width << std::endl;

    for (int i = 0; i < height * width; ++i) {
        if (i < length) {
            input[i / width][i % width] = x_t[i];
        } else {
            input[i / width][i % width] = 0.0; // pad with zeros
        }
    }

    // Call the forward method of the neural network
    std::vector<std::vector<std::vector<double>>> raw_result = nn.forward(input); // Assuming nn.forward returns a 3D vector of doubles

    // Debug print raw_result dimensions after forward
    if (!raw_result.empty() && !raw_result[0].empty()) {
        std::cout << "EpsilonPredictor::predictEpsilon raw_result size: " << raw_result.size() << " x " << raw_result[0].size() << " x " << raw_result[0][0].size() << std::endl;
    } else {
        std::cout << "EpsilonPredictor::predictEpsilon raw_result is empty or malformed" << std::endl;
    }

    if (raw_result.empty() || raw_result[0].empty() || raw_result[0][0].empty()) {
        return std::vector<int>(); // Return empty vector if output is empty to avoid length errors
    }

    int dim1 = raw_result.size(); // Get the first dimension size
    int dim2 = raw_result[0].size(); // Get the second dimension size
    int dim3 = raw_result[0][0].size(); // Get the third dimension size

    // Debug print dimensions
    std::cout << "EpsilonPredictor::predictEpsilon raw_result dimensions: " << dim1 << " x " << dim2 << " x " << dim3 << std::endl;

    // Flatten the 3D result and convert to integers
    std::vector<int> result(dim1 * dim2 * dim3); // Preallocate the result vector
    for (int i = 0; i < dim1; ++i) {
        for (int j = 0; j < dim2; ++j) {
            for (int k = 0; k < dim3; ++k) {
                int flat_idx = i * dim2 * dim3 + j * dim3 + k; // Calculate the flat index
                result[flat_idx] = static_cast<int>(raw_result[i][j][k]); // Convert to int and store in the result vector
            }
        }
    }
    return result;
};
