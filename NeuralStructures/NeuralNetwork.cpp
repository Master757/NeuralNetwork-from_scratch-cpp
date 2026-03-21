#include "NeuralNetwork.h"
#include <iostream>
#include <vector>
#include <math.h>

double sig(double v){
    return 1.0/ (1.0 + exp(-v));
}

double Dsig(double v){//need this for back prop
    return v*(1 - v);
}

//activation function
double RELU(double v){
    if (v > 0){
        v = v;
    }else{
        v = 0;
    }

    return v;
}

//derivative of activtion function (for back prop)
double dRELU(double v){
    return (v > 0) ? 1: 0;
}

NeuralNetwork::NeuralNetwork(std::vector<int> t, double l) : topo(t), learningRate(l){
    for (int i = 0; i < topo.size(); i++){

        //Creating the neuron layers for base
        Matrix n(topo[i], 1); // for a basic Xor, the first would be 2
        NeuronLayer.push_back(n);

        //adding a weightage layer
        if (i < (topo.size() - 1)){
            //basically after every later except the last layer
            Matrix w(topo[i+1], topo[i]);
            w.randomize();
            WeightLayer.push_back(w);
        }

        //now adding the bias layers

        if (i < (topo.size() - 1)){
            //again, we have to skip the last layter there is no bias
            //given after the output is already out
            Matrix b(topo[i + 1], 1);
            b.randomize();
            BiasLayer.push_back(b);
        }
    }

}

void NeuralNetwork::forward(std::vector<double> inputs){
    //this will take the the values from the input later
    //and then push it throuh to the output later

    for (int i = 0; i < inputs.size(); i++) {
        NeuronLayer[0].set(i, 0, inputs[i]);
        //basically put the first layer as input later
    }

    for (int i = 0; i < WeightLayer.size(); i++) {
        Matrix z = WeightLayer[i].mul(NeuronLayer[i]);
        z = z.add(BiasLayer[i]);

        
        NeuronLayer[i + 1] = z.applyFunc(sig);
    }
}

void NeuralNetwork::backward(std::vector<double> outputs){
    //reverse engeneering everything, of the backward feed
    Matrix target(topo.back(), 1);
    for(int i = 0; i < outputs.size(); i++){
        target.set(i, 0, outputs[i]);
    }
//calculations
    Matrix e = target.subtract(NeuronLayer.back());

    for (int i = WeightLayer.size() - 1; i >= 0; i--) {
        Matrix gradients = NeuronLayer[i + 1].applyFunc(Dsig);

        //calculating adjusted gradients
        gradients = gradients.mulElement(e);

        gradients = gradients.applyFunc([&](double x) { 
            return x * learningRate; 
        });

        Matrix prevLayerTransposed = NeuronLayer[i].transpose();
        Matrix weightGradients = gradients.mul(prevLayerTransposed);

        //weight adjustment
        WeightLayer[i] = WeightLayer[i].add(weightGradients);
        BiasLayer[i]   = BiasLayer[i].add(gradients);

        if (i > 0) {
            Matrix weightsTransposed = WeightLayer[i].transpose();
            e = weightsTransposed.mul(e);
        }
    }
}
