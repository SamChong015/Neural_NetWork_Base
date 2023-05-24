#pragma once

#include "Matrix.h"

#include <vector>
#include <cassert>
#include <random>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>

// Random Number
double randomNum()
{
    std::random_device rd;
    std::default_random_engine engine(rd());

    // Create a uniform real distribution between 0 and 1
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    // Generate a random number between 0 and 1
    double random_number = distribution(engine);

    return random_number;
}

class NetWork
{
public:
    NetWork(std::vector<int> top, std::string fileName, bool file = true, double dropout = 0.5, double regularization = 0.01);

    void FeedForward();
    double Error();
    void BackPropagateWithRegularization();
    double L2Regularization();
    void Dropout(Matrix& matrix);
    double Validate();
    void Train(int numEpochs, double learningRate, std::string fileName);

    void setInput(const std::vector<double>& input);
    void setOutput(const std::vector<double>& output);
    void setValidationInput(const std::vector<double>& validationInput);
    void setValidationOutput(const std::vector<double>& validationOutput);
    std::vector<double> getOutput() const;

private:
    std::vector<int> topology;
    std::vector<Matrix> values;
    std::vector<Matrix> weights;
    std::vector<Matrix> prevWeights;
    std::vector<Matrix> biass;
    std::vector<Matrix> errors;
    std::vector<double> input;
    std::vector<double> output;
    std::vector<double> validationInput;
    std::vector<double> validationOutput;
    double dropoutProb;
    double lambda;

    // Read network from file
    void readNetworkFromFile(const std::string& filename)
    {
        std::ifstream file(filename);
        if (!file) {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        std::string line;
        std::getline(file, line); // Read the comment

        // Read topology
        int topologySize = 0;
        file >> topologySize;
        file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        std::vector<int> topology(topologySize);
        for (int i = 0; i < topologySize; ++i) {
            file >> topology[i];
        }

        std::getline(file, line); // Read the empty line before each values section
        
        // Read values
        for (int i = 0; i < topologySize; ++i) {
            std::getline(file, line); // Read the empty line before each values section
            std::getline(file, line); // Read the empty line before each values section
            values[i].readFromFile(file);
            std::getline(file, line); // Read the empty line before each values section
        }

        // Read weights
        for (int i = 0; i < topologySize - 1; ++i) {
            std::getline(file, line); // Read the empty line before each weights section
            std::getline(file, line); // Read the empty line before each values section
            weights[i].readFromFile(file);
            std::getline(file, line); // Read the empty line before each values section
        }

        // Read biases
        for (int i = 0; i < topologySize - 1; ++i) {
            std::getline(file, line); // Read the empty line before each values section
            std::getline(file, line); // Read the empty line before each biases section
            biass[i].readFromFile(file);
            std::getline(file, line); // Read the empty line before each values section
        }
        file.close();
    }

    // Write network to file
    void writeToFile(const std::string& filename) const
    {
        std::ofstream file(filename);
        if (!file) {
            throw std::runtime_error("Failed to create file: " + filename);
        }

        displayNetwork();

        // Write topology
        int topologySize = topology.size();
        file << "# Topology" << std::endl;
        file << topologySize << "# Size of Topology" << std::endl;
        for (int i = 0; i < topologySize; ++i) {
            file << topology[i] << " ";
        }
        file << std::endl;

        // Write values
        for (int i = 0; i < topologySize; ++i) {
            file << std::endl << "# Values Layer " << i << std::endl;
            values[i].writeToFile(file); // Pass the file stream to the Matrix's writeToFile function
        }

        // Write weights
        for (int i = 0; i < topologySize - 1; ++i) {
            file << std::endl << "# Weights Layer " << i << " to " << i + 1 << std::endl;
            weights[i].writeToFile(file); // Pass the file stream to the Matrix's writeToFile function
        }

        // Write biases
        for (int i = 0; i < topologySize - 1; ++i) {
            file << std::endl << "# Biases Layer " << i + 1 << std::endl;
            biass[i].writeToFile(file); // Pass the file stream to the Matrix's writeToFile function
        }

        file.close();
    }

    // Display network on console
    void displayNetwork() const
    {
        // Write topology
        int topologySize = topology.size();
        std::cout << "# Topology" << std::endl;
        std::cout << topologySize << " # Size of topology" << std::endl;
        for (int i = 0; i < topologySize; ++i) {
            std::cout << topology[i] << " ";
        }
        std::cout << std::endl;
        
        // Write values
        for (int i = 0; i < topologySize; ++i) {
            std::cout << std::endl << "# Values Layer " << i << std::endl;
            values[i].display();
        }

        // Write weights
        for (int i = 0; i < topologySize - 1; ++i) {
            std::cout << std::endl << "# Weights Layer " << i << " to " << i + 1 << std::endl;
            weights[i].display();
        }

        // Write biases
        for (int i = 0; i < topologySize - 1; ++i) {
            std::cout << std::endl << "# Biases Layer " << i + 1 << std::endl;
            biass[i].display();
        }
    }
};

NetWork::NetWork(std::vector<int> top, std::string fileName, bool file, double dropout, double regularization)
    : topology(top), dropoutProb(dropout), lambda(regularization)
{
    for (int i = 0; i < topology.size() - 1; i++)
    {
        Matrix val(1, topology.at(i));
        Matrix weight(topology.at(i), topology.at(i + 1));

        for (int h = 0; h < topology.at(i); h++) // randomly sets weights
        {
            for (int j = 0; j < topology.at(i + 1); j++)
            {
                weight.setData(h, j, randomNum());
            }
        }
        Matrix bias(1, topology.at(i + 1));
        Matrix error(1, topology.at(i + 1));

        values.push_back(val);
        weights.push_back(weight);
        prevWeights.push_back(weight);
        biass.push_back(bias);
        errors.push_back(error);
    }
    Matrix val(1, topology.back());

    values.push_back(val);

    if (file == true)
    {
        readNetworkFromFile(fileName);
    }
   
}

void NetWork::setInput(const std::vector<double>& input) 
{
    for (size_t i = 0; i < input.size(); ++i) {
        values.front().setData(0, i, input[i]);
    }
}

void NetWork::setOutput(const std::vector<double>& out) 
{
    this->output = out;
}

void NetWork::setValidationInput(const std::vector<double>& va) 
{
    validationInput = va;
}

void NetWork::setValidationOutput(const std::vector<double>& va) 
{
    validationOutput = va;
}

std::vector<double> NetWork::getOutput() const {
    std::vector<double> result;

    Matrix temp = values.back();

    for (int i = 0; i < temp.getColumns(); ++i) {
        result.push_back(values.back().getData(0, i));
    }

    return result;
}

void NetWork::FeedForward()
{
    for (int i = 0; i < topology.size() - 1; i++)
    {
        values.at(i + 1) = values.at(i) * weights.at(i) + biass.at(i);

        values.at(i).sigmoidTransformation();

        // Apply dropout to hidden layers (not applied to input and output layers)
        if (i > 0 && i < topology.size() - 2)
            Dropout(values.at(i));
    }
}

double NetWork::Error()
{
    double error = 0.0;
    for (size_t i = 0; i < values.back().getColumns(); ++i)
    {
        // Add a small constant value to avoid taking the logarithm of zero
        double epsilon = 1e-10;
        error += output[i] * log(values.back().getData(0, i) + epsilon);
    }

    return -error;
}

void NetWork::BackPropagateWithRegularization()
{
    // Calculate the errors in the output layer
    for (int i = 0; i < topology.back(); i++)
    {
        errors.back().setData(0, i, output[i] - values.back().getData(0, i));
    }

    // Backpropagate the errors through the network
    for (int i = topology.size() - 3; i >= 0; i--)
    {
        Matrix transposedWeights = weights[i].transpose();

        Matrix temp = errors[i+1];

        errors[i] = temp * weights[i]; //origionally was transposed
    }

    // Update the weights and biases using the calculated errors
    for (int i = 0; i < topology.size() - 1; i++)
    {
        Matrix deltaWeights = values[i].transpose() * errors[i];

        weights[i] = weights[i] + deltaWeights - (weights[i] * lambda); // Regularization term

        // Update biases
        Matrix deltaBiases = errors[i];
        biass[i] = biass[i] + deltaBiases;
    }
}

double NetWork::L2Regularization()
{
    double regularization = 0.0;

    for (const auto& weight : weights)
    {
        Matrix squaredWeights = weight.multiplyElementwise(weight);
        regularization += squaredWeights.sum();
    }

    return 0.5 * lambda * regularization;
}

void NetWork::Dropout(Matrix& matrix)
{
    std::random_device rd;
    std::default_random_engine engine(rd());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < matrix.getRows(); i++)
    {
        for (int j = 0; j < matrix.getColumns(); j++)
        {
            if (distribution(engine) < dropoutProb)
                matrix.setData(i, j, 0.0);
        }
    }
}

double NetWork::Validate()
{
    // Set the network input to the validation input
    for (size_t i = 0; i < input.size(); ++i)
    {
        values.front().setData(0, i, validationInput[i]);
    }

    // Set the expected output to the validation output
    output = validationOutput;

    // Feedforward
    FeedForward();

    // Calculate error
    double error = Error();

    return error;
}

void NetWork::Train(int numEpochs, double learningRate, std::string fileName)
{
    double bestValidationError = std::numeric_limits<double>::max();
    int bestEpoch = 0;

    for (int epoch = 0; epoch < numEpochs; ++epoch)
    {
        // Forward propagation
        FeedForward();

        // Perform backpropagation with regularization
        BackPropagateWithRegularization();

        // Update weights and biases
        for (int i = 0; i < topology.size() - 1; i++)
        {
            Matrix deltaWeights = values[i].transpose() * errors[i];
            
            weights[i] = weights[i] + deltaWeights * learningRate;
           
            // Update biases
            Matrix deltaBiases = errors[i];
           
            biass[i] = biass[i] + deltaBiases * learningRate;
        }

        // Perform validation
        double validationError = Validate();

        if (validationError < bestValidationError)
        {
            bestValidationError = validationError;
            bestEpoch = epoch;
        }

        // Early stopping condition
        if (epoch - bestEpoch >= 10) { // Stop training if no improvement after 10 epochs
            std::cout << "Early stopping triggered. Best validation error: " << bestValidationError << std::endl;
            break;
        }
    }

    writeToFile(fileName);
}
