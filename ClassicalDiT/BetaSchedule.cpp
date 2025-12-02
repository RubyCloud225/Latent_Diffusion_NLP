#include "BetaSchedule.hpp"

//Constructor
BetaSchedule::BetaSchedule(int total_epochs, double initial_beta) : total_epochs_(total_epochs), initial_beta_(initial_beta), current_beta_(initial_beta) {}

// update method to calculate the current beta based on losses
double BetaSchedule::update(const std::vector<double>& nll_losses, const std::vector<double>& entropy_losses, int epoch) {
    if (epoch < 0 || epoch >= total_epochs_) {
        std::cerr << "Epoch out of range." << std::endl;
        return current_beta_;
    }
    // Calculate the current beta value
    double beta_t = initial_beta_ * (1.0 - static_cast<double>(epoch) / total_epochs_) + (static_cast<double>(epoch) / total_epochs_);
    current_beta_ = beta_t;
    return current_beta_;
}

// get method to retrieve the current beta value
double BetaSchedule::getCurrentBeta() const {
    return current_beta_;
}

// get the current beta value
double BetaSchedule::getInitialBeta() const {
    return initial_beta_;
}

// get number of epochs
int BetaSchedule::getNumEpochs() const {
    return total_epochs_;
}