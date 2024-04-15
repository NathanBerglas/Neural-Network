#include <stdlib.h>
#include <stdio.h>
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

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double activation(double x) {
    // Sigmoid
    return sigmoid(x);
}

// required for opaque structure
int inputsNN(struct neuralNetwork *nn) {
    return nn->input->neuronCount;
}

// Forward Propagation
void runNN(struct neuralNetwork *nn, double *inputData) {
    // Starts at input. Assigns input neurons given input
    for (int n = 0; n < nn->input->neuronCount; n++) {
        nn->input->neurons[n]->output = inputData[n];
    }

    // Propogates through the first layer with input from input
    for (int n = 0; n < nn->hidden[0]->neuronCount; n++) {
        nn->hidden[0]->neurons[n]->output = nn->hidden[0]->neurons[n]->bias; // Start output at bias
        for (int i = 0; i < nn->input->neuronCount; i++) {
            nn->hidden[0]->neurons[n]->output += nn->input->neurons[i]->output * nn->input->neurons[i]->weights[n]; // Add each input from previous nodes to output
        } // Results in output = bias + sum of inputs from previous nodes * weights
        nn->hidden[0]->neurons[n]->output = activation(nn->hidden[0]->neurons[n]->output); // Runs activation function
    }

    // Propogates through hidden layers until output
    // Incredibly similar as last bit of code, just now takes input from last hidden layer, not input
    for (int l = 1; l < nn->hiddenLayerCount; l++) {
        for (int n = 0; n < nn->hidden[l]->neuronCount; n++) {
            nn->hidden[l]->neurons[n]->output = nn->hidden[l]->neurons[n]->bias;
            for (int i = 0; i < nn->hidden[l-1]->neuronCount; i++) {
                nn->hidden[l]->neurons[n]->output += nn->hidden[l-1]->neurons[i]->output * nn->hidden[l-1]->neurons[i]->weights[n];
            }
            nn->hidden[l]->neurons[n]->output = activation(nn->hidden[l]->neurons[n]->output); // Runs activation function
        }
    }

    // Propogates from final hidden layer to output
    // Still same code
    for (int n = 0; n < nn->output->neuronCount; n++) {
        nn->output->neurons[n]->output = nn->output->neurons[n]->bias;
        for (int i = 0; i < nn->hidden[nn->hiddenLayerCount - 1]->neuronCount; i++) {
            nn->output->neurons[n]->output += nn->hidden[nn->hiddenLayerCount - 1]->neurons[i]->output * nn->hidden[nn->hiddenLayerCount - 1]->neurons[i]->weights[n];
        }
        nn->output->neurons[n]->output = activation(nn->output->neurons[n]->output); // Runs activation function
    }
}

void printNN(struct neuralNetwork *nn) {
    printf("Inputs: ");
    for (int i = 0; i < nn->input->neuronCount; i++) {
        if (i + 1 != nn->input->neuronCount) {
            printf("%lf, ", nn->input->neurons[i]->output);
        } else { 
            printf("%lf\nOutputs: ", nn->input->neurons[i]->output);
        }
    }
    for (int o = 0; o < nn->output->neuronCount; o++) {
        if (o + 1 != nn->output->neuronCount) {
            printf("%lf, ", nn->output->neurons[o]->output);
        } else {
            printf("%lf\n", nn->output->neurons[o]->output);
        }
    }
}

void trainNN(struct neuralNetwork *nn, const double lr, int epochs, double *trainingInputs, double *trainingOutputs) {

}


// Weights & Biases tools
void asignWeights(struct neuralNetwork *nn, double* weights) {

}

void exportWeights(struct neuralNetwork *nn) {

}


