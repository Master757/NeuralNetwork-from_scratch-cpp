#include <iostream>
#include <vector>
#include "NeuralStructures/NeuralNetwork.h"
#include "DatasetManagement/DatasetLoader.cpp" // Ensure the path is correct

int main() {
    int ep = 8000;
    DatasetLoader loader;
    loader.loadFromCSV("archive/IRIS.csv", 4, 3);
    loader.shuffle();

    // Split the data
    auto testSet = loader.getTestBatch(0.2); 
    auto trainingSet = loader.getData();     // This returns std::vector<DataInstance>

    std::vector<int> topology = {4, 8, 3};
    NeuralNetwork nn(topology, 0.01);

    std::cout << "--- Starting Training (Iris-Dataset) ---" << std::endl;

    for (int epoch = 0; epoch < ep; epoch++) {
        //loader.shuffle();
        for (const auto& data : trainingSet) {
            nn.forward(data.inputs);   // Using 'inputs' from DataInstance
            nn.backward(data.targets); // Using 'targets' from DataInstance
        }

        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << " complete...." << std::endl;
        }
    }
   std::cout << "\n--- Testing Phase ---" << std::endl;

    int correct = 0;

    for (const auto& data : testSet) {
        nn.forward(data.inputs);
        
        // The output layer is the last matrix in NeuronLayer
        Matrix output = nn.NeuronLayer.back();
        
        int predictedLabel = 0;
        double maxVal = -1.0;
        int actualLabel = 0;

        // Finding the index of the highest output (The Prediction)
        for (int j = 0; j < output.getRows(); j++) {
            if (output.get(j, 0) > maxVal) {
                maxVal = output.get(j, 0);
                predictedLabel = j;
            }
        }

        // Find the index of the 1.0 in targets (The Truth)
        for (int j = 0; j < data.targets.size(); j++) {
            if (data.targets[j] == 1.0) {
                actualLabel = j;
                break;
            }
        }

        if ((predictedLabel == actualLabel)) {
            correct++;
        }
    }

    double accuracy = (double)correct / testSet.size() * 100.0;
    std::cout << "Final Test Accuracy: " << accuracy << "% (" 
            << correct << "/" << testSet.size() << " correct)" << std::endl;
    return 0;
}