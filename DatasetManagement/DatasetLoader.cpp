#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <iostream>

struct DataInstance {
    std::vector<double> inputs;
    std::vector<double> targets;
};

class DatasetLoader {
private:
    std::vector<DataInstance> data;

public:
    void loadFromCSV(std::string filename, int numInputs, int numOutputs, bool hasIdColumn = false) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return;
        }

        std::string line;
        bool isHeader = true;

        while (std::getline(file, line)) {
            if (line.empty()) continue;
            if (isHeader) { isHeader = false; continue; }

            std::stringstream ss(line);
            std::string value;
            DataInstance instance;

            // 1. Skip the ID column if it exists
            if (hasIdColumn) {
                std::getline(ss, value, ',');
            }

            // 2. Load Inputs
            for (int i = 0; i < numInputs; i++) {
                if (std::getline(ss, value, ',')) {
                    try {
                        instance.inputs.push_back(std::stod(value));
                    } catch (...) {
                        instance.inputs.push_back(0.0); // Fallback for bad data
                    }
                }
            }

            // 3. Load and Map Target
            if (std::getline(ss, value, ',')) {
                instance.targets.resize(numOutputs, 0.0);
                // Cleans up stray quotes or spaces from the string
                value.erase(std::remove(value.begin(), value.end(), '\"'), value.end());
                
                if (value == "Iris-setosa" || value == "0") instance.targets[0] = 1.0;
                else if (value == "Iris-versicolor" || value == "1") instance.targets[1] = 1.0;
                else if (value == "Iris-virginica" || value == "2") instance.targets[2] = 1.0;
            }

            if (instance.inputs.size() == numInputs) {
                data.push_back(instance);
            }
        }
        file.close();
        std::cout << "Successfully loaded " << data.size() << " instances." << std::endl;
    }

    // Normalizing values to 0.0 - 1.0 helps the Neural Network learn faster
    void normalize() {
        if (data.empty()) return;
        int inputSize = data[0].inputs.size();

        for (int i = 0; i < inputSize; i++) {
            double min = data[0].inputs[i];
            double max = data[0].inputs[i];

            for (auto& instance : data) {
                if (instance.inputs[i] < min) min = instance.inputs[i];
                if (instance.inputs[i] > max) max = instance.inputs[i];
            }

            for (auto& instance : data) {
                if (max - min != 0) {
                    instance.inputs[i] = (instance.inputs[i] - min) / (max - min);
                }
            }
        }
    }

    void shuffle() {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(data.begin(), data.end(), g);
    }

    std::vector<DataInstance> getTestBatch(float percentage) {
        int testSize = static_cast<int>(data.size() * percentage);
        std::vector<DataInstance> testData(data.end() - testSize, data.end());
        data.erase(data.end() - testSize, data.end());
        return testData;
    }

    const std::vector<DataInstance>& getData() const { return data; }
};