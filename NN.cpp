/*this is a cpp program to construct a neural 
network. This is to help understand better 
the working definitions and algorithms as 
similarly present, in an actual neural
network as we do with the help of TensorFlow
or pyTorch.*/

#include <iostream>
#include <vector>
#include "NeuralStructures/NeuralNetwork.h"

int main() {
    struct TrainingData {
        std::vector<double> input;
        std::vector<double> target;
    };

    std::vector<TrainingData> trainingSet = {
        { {0.0, 0.0}, {0.0} },
        { {0.0, 1.0}, {1.0} },
        { {1.0, 0.0}, {1.0} },
        { {1.0, 1.0}, {0.0} }
    };

    std::vector<int> topology = {2, 3, 1};
    NeuralNetwork nn(topology, 0.1); // Learning Rate = 0.1

    std::cout << "--- Starting Training (XOR Problem) ------" << std::endl;

    for (int epoch = 0; epoch < 10000; epoch++) {
        
        for (int i = 0; i < trainingSet.size(); i++) {
            nn.forward(trainingSet[i].input);
            nn.backward(trainingSet[i].target);
        }

        // printing the values so that we know that it is indeed training
        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << " complete..." << std::endl;
        }
    }

    std::cout << "\n----- Training Complete! Final Results ---" << std::endl;

    for (int i = 0; i < trainingSet.size(); i++) {
        nn.forward(trainingSet[i].input);
        
        std::cout << "Input: " << trainingSet[i].input[0] << ", " << trainingSet[i].input[1] << " | Target: " << trainingSet[i].target[0] << " | Output: ";
               
        nn.NeuronLayer.back().print(); 
    }

    return 0;
}