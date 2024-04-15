#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "neuralNetwork.h"

struct neuron {
    double *weights;
    double bias;
    double output;
};

struct layer {
    struct neuron **neurons;
    int neuronCount;
};

struct neuralNetwork {
    struct layer *input;
    struct layer** hidden;
    int hiddenLayerCount;
    struct layer *output;
};

//Intialization
double weightInit() {
    return ((double)rand() / RAND_MAX) * 2 - 1; // From -1 to 1
}

double biasInit() {
    return ((double)rand() / RAND_MAX) * 0.01 - 0.005; // Small value between -0.005 to 0.005
}

struct neuron *nInit(int neuronCountNextLayer) {
    struct neuron *n = malloc(sizeof(struct neuron));
    n->weights = malloc(neuronCountNextLayer * sizeof(double));
    for (int nextn = 0; nextn < neuronCountNextLayer; nextn++) {
        n->weights[nextn] = weightInit();
    }
    n->bias = weightInit();
    return n;
}

struct layer *layerInit(int neuronCount, int neuronCountNextLayer) {
    struct layer *l = malloc(sizeof(struct layer));
    l->neurons = malloc(neuronCount * sizeof(struct neuron*));
    assert(l);
    for (int n = 0; n < neuronCount; n++) {
        l->neurons[n] = nInit(neuronCountNextLayer);
    }
    l->neuronCount = neuronCount;
    return l;
}

struct neuralNetwork *nnInit(int inputNeuronCount, int hiddenLayerCount, int *hiddenLayerNeuronCount, int outputNeuronCount) {
    struct neuralNetwork *nn = malloc(sizeof(struct neuralNetwork));
    assert(nn);
    nn->input = layerInit(inputNeuronCount, hiddenLayerNeuronCount[0]);
    nn->hidden = malloc(hiddenLayerCount * sizeof(struct layer*));
    for (int hl = 0; hl < hiddenLayerCount; hl++) {
        if (hl + 1 == hiddenLayerCount) {
            nn->hidden[hl] = layerInit(hiddenLayerNeuronCount[hl], outputNeuronCount);
        } else {
            nn->hidden[hl] = layerInit(hiddenLayerNeuronCount[hl], hiddenLayerNeuronCount[hl + 1]);
        }
    }
    nn->hiddenLayerCount = hiddenLayerCount;
    nn->output = layerInit(outputNeuronCount, 0);
    return nn;
}

void nFree(struct neuron* n) {
    assert(n);
    free(n->weights);
    free(n);
}

void lFree(struct layer* l) {
    assert(l);
    for(int n = 0; n < l->neuronCount; n++) {
        nFree(l->neurons[n]);
    }
    free(l->neurons);
    free(l);
}

void nnFree(struct neuralNetwork *nn) {
    assert(nn);
    lFree(nn->input);
    for (int l = 0; l < nn->hiddenLayerCount; l++) {
        lFree(nn->hidden[l]);
    }
    free(nn->hidden);
    lFree(nn->output);
    free(nn);
}

// NN tools
void runNN(struct neuralNetwork *nn, double *input) {

}

void trainNN(struct neuralNetwork *nn, const double lr, int epochs, double *trainingInputs, double *trainingOutputs) {

}


// Weights & Biases tools
void asignWeights(struct neuralNetwork *nn, double* weights) {

}

void exportWeights(struct neuralNetwork *nn) {

}



