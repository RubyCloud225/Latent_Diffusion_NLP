#ifndef BETASCHEDULE_HPP
#define BETASCHEDULE_HPP

#include <vector>
#include <random>
#include <iostream>

class BetaSchedule {
    public:
    BetaSchedule(int total_epochs, double initial_beta);
    // Method to update losses and calculate the current beta
    double update(const std::vector<double>& nll_losses, const std::vector<double>& entropy_losses, int epoch);
    // Get current beta value
    double getCurrentBeta() const;
    // Get the number of epochs
    int getNumEpochs() const;
    // Get the initial beta value
    double getInitialBeta() const;
    private:
    int total_epochs_;
    double initial_beta_;
    double current_beta_;
};

#endif