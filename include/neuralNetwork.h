// layer.h
#pragma once

struct neuralNetwork;

// Initialization
struct neuralNetwork *nnInit(int inputNeuronCount, int hiddenLayerCount, int *hiddenLayerNeuronCount, int outputNeuronCount);
void nnFree(struct neuralNetwork *nn);

// NN Tools
void runNN(struct neuralNetwork *nn, double *input);
void trainNN(struct neuralNetwork *nn, const double lr, int epochs, double *trainingInputs, double *trainingOutputs);

// Weights & Biases tools
void asignWeights(struct neuralNetwork *nn, double* weights);
void exportWeights(struct neuralNetwork *nn);