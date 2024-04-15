// layer.h
#pragma once

struct neuralNetwork;

// Initialization
struct neuralNetwork *nnInit(int inputNeuronCount, int hiddenLayerCount, int *hiddenLayerNeuronCount, int outputNeuronCount);
void nnFree(struct neuralNetwork *nn);

// NN Tools
int inputsNN(struct neuralNetwork *nn);
void runNN(struct neuralNetwork *nn, double *input);
void printNN(struct neuralNetwork *nn);
void trainNN(struct neuralNetwork *nn, const double lr, int epochs, double *trainingInputs, double *trainingOutputs);

// Weights & Biases tools
void assignWeights(struct neuralNetwork *nn, FILE *file);
void exportWeights(struct neuralNetwork *nn, FILE *file);