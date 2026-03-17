# Raw Neural Networks (C++)

## Overview

This project demonstrates a mathematical, from-scratch implementation of a neural network in C++, built without using any machine learning libraries.

The goal of this project is to show that neural networks can be constructed purely from linear algebra operations such as matrix multiplication, addition, transposition, and element-wise functions, closely mirroring the mathematical foundations of neural networks.

The implementation is intentionally low-level and explicit to emphasize understanding over abstraction.

---

## Project Structure

RawNeuralNetworks/
|
|-- Matrix/
|   |-- Matrix.h
|   |-- Matrix.cpp
|
|-- NeuralStructures/
|   |-- NeuralNetwork.h
|   |-- NeuralNetwork.cpp
|
|-- NN.cpp
|-- README.md

---

## Matrix Module

The Matrix module acts as the mathematical backbone of the project.

It provides:
- Matrix multiplication
- Element-wise operations (Hadamard product)
- Addition and subtraction
- Transpose
- Function application (used for activations)
- Random initialization (for weights and biases)

All neural network computations are built entirely on top of this module.

---

## Neural Network Module

The NeuralNetwork module builds a fully connected feedforward neural network using the Matrix module.

Features:
- Configurable network topology
- Forward propagation
- Backpropagation using gradient descent
- Bias and weight updates
- Sigmoid activation function
- Dataset extraction

The implementation follows the standard mathematical formulation:

Z = W · A + B  
A = σ(Z)

and applies backpropagation through explicit matrix operations.

---

## Demonstration: XOR Problem

This version of the project is validated using the XOR problem, a classic benchmark that demonstrates the necessity of hidden layers in neural networks.

While the example focuses on XOR, the architecture supports:
- Arbitrary network depth
- Arbitrary layer sizes
- Extension to other problems by adjusting topology and training logic

---

## Design Philosophy

- No external ML libraries
- Explicit math over abstraction
- Clear separation between math and network logic
- Educational and extensible by design

This project prioritizes clarity and correctness of the underlying mathematics rather than performance or convenience.

---

## Current Limitations (Version 1)

- Single activation function (Sigmoid)
- Basic gradient descent optimizer
- No batching or dataset loader
- No automatic differentiation
- Designed primarily for educational and experimental use

These limitations are intentional for Version 1.

---

## Future Work

Planned improvements include:
- Additional activation functions (ReLU, Tanh)
- Cleaner backpropagation flow
- Modular loss functions
- Optimizers beyond basic gradient descent
- Better error handling and validation
- Potential integration with a custom programming language runtime

---

## Why This Project Exists

Most modern ML frameworks abstract away the mathematics behind neural networks.

This project exists to expose and understand those mechanics directly, reinforcing how neural networks operate at a fundamental level.

---

## Build & Run

Compile using a standard C++ compiler:

g++ -O3 NN.cpp Matrix/Matrix.cpp NeuralStructures/NeuralNetwork.cpp DatasetManagement/DatasetLoader.cpp -o neuralnet

Then run:

./neuralnet

---

##Recent Changes
Adding iris Dataset for the prediction. The parameters are not finetuned and thus the outputs are not *upto mark*. A dataset extraction was added and is made as dynamic as possible. But reviewing the dataset *needs* to be be done on dataset. One can not perform machine Learning without understanding the dataset.
---

## Final Notes
This project is Still a learning opportunity about Neural nets on a mathamatical level
The parameters will be finetuned, for better and more stable results.
