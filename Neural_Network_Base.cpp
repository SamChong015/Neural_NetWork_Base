// Neural_Network_Base.cpp : This file contains the 'main' function. 

#include <iostream>

#include "NetWork.h"
#include <vector>

int main() { // doesn't work with inputing from a file
    int hiddenLayers;
    int inputs;
    int outputs;

    std::string res;
    std::string fileName;

    bool file = true;
    double dropout;
    double regularization;

    std::vector<int> topology;

    std::cout << "Load from file? Y or N ";
    std::cin >> res;
    std::cout << '\n';

    if (res == "N" || res == "n")
    {
        file = false;
        std::cout << "How many input nodes? ";
        std::cin >> inputs;

        topology.push_back(inputs);
        
        std::cout << '\n' << "How many hidden layers? ";
        std::cin >> hiddenLayers;
        for (int i = 0; i < hiddenLayers; i++)
        {
            int hid;
            std::cout << '\n' << "How many nodes in hidden layer? " << i + 1 << "?";
            std::cin >> hid;
            topology.push_back(hid);
        }

        std::cout << '\n' << "How many output nodes? ";
        std::cin >> outputs;
        topology.push_back(outputs);
    }
    
    std::cout << '\n' << "What is your dropout value? eg 0.5 ";
    std::cin >> dropout;

    std::cout << '\n' << "What is your regularization factor? eg 0.01 ";
    std::cin >> regularization;

    std::cout << '\n' << "What is your file name? ";
    std::cin >> fileName;

    std::cout << '\n' << "Creating Network... ";
    NetWork network(topology, fileName, file, dropout, regularization);
    std::cout << " ...Network Created" << '\n';

    double learningRate;
    std::string trainingDataFile;
    std::string verificationDataFile;

    std::cout << "What learning rate would you like? eg 0.01 ";
    std::cin >> learningRate;

    std::cout << '\n' << "What is the training data file name? ";
    std::cin >> trainingDataFile;

    std::cout << '\n' << "What is the verification data file name? ";
    std::cin >> verificationDataFile;

    std::cout << '\n' << "Beginning Training... ";
    network.Train(learningRate, fileName, trainingDataFile, verificationDataFile);
    std::cout << "...Training Finished" << '\n';

    // Get the output of the trained network
    std::vector<double> networkOutput = network.getOutput();

    // Print the network output
    std::cout << "Network Output: ";
    for (double value : networkOutput) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    network.writeToFile(fileName);

    return 0;
}
