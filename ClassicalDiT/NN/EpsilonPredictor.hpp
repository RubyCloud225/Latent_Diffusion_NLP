#ifndef EPSILONPREDICTOR_HPP
#define EPSILONPREDICTOR_HPP

#include "NeuralNetwork.hpp"
#include <iostream>
#include <vector>

class EpsilonPredictor {
    public:
    EpsilonPredictor(int input_channels, int output_size);
    std::vector<int> predictEpsilon(const std::vector<double>& x_t, int t);
    private:
    NeuralNetwork nn_;
};

#endif // EPSILONPREDICTOR_HPP
