#include <iostream>
#include <vector>
#include "../Matrix/Matrix.h"
#pragma once

class NeuralNetwork{
    public:
        std::vector<int> topo;
        double learningRate;
        std::vector<Matrix> NeuronLayer;
        std::vector<Matrix> WeightLayer;
        std::vector<Matrix> BiasLayer;

        //functions and constructor

        NeuralNetwork(std::vector<int> t, double l);

        void forward(std::vector<double> inputs);

        void backward(std::vector<double> outputs);

        void printBias();
        void printNeurons();
};