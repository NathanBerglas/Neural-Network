// layer.h
#pragma once

struct neuralNetwork;

// Initialization

// nnInit: Initializes a new neural network with given parameters and assigns weights and biases randomly
//      Effects:    Allocates memory (Caller must free)
//                  Returns pointer to heap
//      Requires: inputNeuronCount > 0, hiddenLayerCount >= 0, *hiddenLayerNeuronCount > 0, outputNeuronCount > 0
//      Time:   O(n * m^2) Where n is the number of hidden layers and m is the average neuron count per layer
struct neuralNetwork *nnInit(int inputNeuronCount, int hiddenLayerCount, int *hiddenLayerNeuronCount, int outputNeuronCount);

// nnFree: Frees the memory allocated in nn
//      Effects:    Frees memory, renders nn inusable
//      Time:   O(n), n is the number of neurons in a nn
void nnFree(struct neuralNetwork *nn);

// NN Tools

// inputsNN: The number of input nodes in a nn
//      Effects: Returns an integer
//      Time:   O(1)
int inputsNN(struct neuralNetwork *nn);

// outputsNN: The number of output nodes in a nn
//      Effects: Returns an integer
//      Time:   O(1)
int outputsNN(struct neuralNetwork *nn);

// runNN: Runs the neural network with given inputs. Uses forward propogation
//      Requires: size of input is equal ot the input nodes in nn
//      Time: O(n * m^2) Where n is the number of hidden layers and m is the average neuron count per layer
void runNN(struct neuralNetwork *nn, double *input);

// printNN: Prints out the relevant information in a neural network: its inputs and outputs
//      Effects: Writes to terminal
//      Time: O(n + m), n is the number of input nodes and m is the number of output nodes 
void printNN(struct neuralNetwork *nn);

// trainNN: Trains the neural network with the given training data
//      Effects: Mutates nn
//      Requires: lr > 0, epochs > 0, size of *trainingInputs is equal to the number of input nodes, size of *trainingOutputs is equal to the number of output nodes
//      Time: O(???)
void trainNN(struct neuralNetwork *nn, const double lr, int epochs, double **trainingInputs, double **trainingOutputs, int trainingSets);

// Weights & Biases tools

// assignWeights: Mutates the weights and biases nn as described in a binary file
//      Effects: Mutates nn
//      Time: O(n * m^2) Where n is the number of hidden layers and m is the average neuron count per layer
void assignWeights(struct neuralNetwork *nn, FILE *file);

// exportWeights: Outputs to binary file the weights and biases of nn
//      Effects: Writes to file
//      Time: O(n * m^2) Where n is the number of hidden layers and m is the average neuron count per layer
void exportWeights(struct neuralNetwork *nn, FILE *file);