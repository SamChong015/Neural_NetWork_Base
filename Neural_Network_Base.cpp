// Neural_Network_Base.cpp : This file contains the 'main' function. 

/* https://medium.com/coinmonks/the-mathematics-of-neural-network-60a112dd3e05 */

#include <iostream>

#include "NetWork.h"
#include <vector>

int main() {
    // Define the network topology
    std::vector<int> topology = { 2, 4, 2 }; // Input layer with 2 neurons, hidden layer with 4 neurons, output layer with 2 neurons
    std::cout << "Set Topology" << std::endl;

    // Create the network
    NetWork network(topology, "File2", true, 0.5, 0.01); // Randomly initialized weights, dropout probability of 0.5, regularization factor of 0.01
    std::cout << "Network Created..." << std::endl;


    // Set the training data
    std::vector<double> input = { 0.1, 0.2 };
    std::vector<double> output = { 0.3, 0.4 };
    network.setInput(input);
    network.setOutput(output);
    std::cout << "Set Training Data" << std::endl;

    // Set the validation data
    std::vector<double> validationInput = { 0.3, 0.4 };
    std::vector<double> validationOutput = { 0.5, 0.6 };
    network.setValidationInput(validationInput);
    network.setValidationOutput(validationOutput);
    std::cout << "Set Validation Data" << std::endl;

    // Train the network
    int numEpochs = 100;
    double learningRate = 0.01;
    network.Train(numEpochs, learningRate, "File2"); //This is where the code is currently broken
    std::cout << "...Training..." << std::endl;

    // Get the output of the trained network
    std::vector<double> networkOutput = network.getOutput();

    // Print the network output
    std::cout << "Network Output: ";
    for (double value : networkOutput) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    return 0;
}

