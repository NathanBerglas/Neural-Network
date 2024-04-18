#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#include "neuralNetwork.h"

struct neuron {
    double *weights; // Length is the # of neurons in the next layer
    double bias;
    double output;
};

struct layer {
    struct neuron **neurons;
    int neuronCount;
};

struct neuralNetwork {
    struct layer *input;
    struct layer** hidden; // Length is hiddenLayerCount
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
    if (!n) {
        printf("ERROR: Not enough stack memory! n\n");
        assert(0);
    }
    n->weights = malloc(neuronCountNextLayer * sizeof(double));
    if (!n->weights) {
        printf("ERROR: Not enough stack memory! n->weights\n");
        assert(0);
    }
    for (int nextn = 0; nextn < neuronCountNextLayer; nextn++) {
        n->weights[nextn] = weightInit();
    }
    n->bias = weightInit();
    return n;
}

struct layer *layerInit(int neuronCount, int neuronCountNextLayer) {
    struct layer *l = malloc(sizeof(struct layer));
    if (!l) {
        printf("ERROR: Not enough stack memory! l\n");
        assert(0);
    }
    l->neurons = malloc(neuronCount * sizeof(struct neuron*));
    if (!l->neurons) {
        printf("ERROR: Not enough stack memory! l->neurons\n");
        assert(0);
    }
    assert(l);
    for (int n = 0; n < neuronCount; n++) {
        l->neurons[n] = nInit(neuronCountNextLayer);
    }
    l->neuronCount = neuronCount;
    return l;
}

struct neuralNetwork *nnInit(int inputNeuronCount, int hiddenLayerCount, int *hiddenLayerNeuronCount, int outputNeuronCount) {
    struct neuralNetwork *nn = malloc(sizeof(struct neuralNetwork));
    if (!nn) {
        printf("ERROR: Not enough stack memory! nn\n");
        assert(0);
    }
    assert(inputNeuronCount > 0);
    assert(hiddenLayerCount >= 0);
    assert(outputNeuronCount > 0);
    nn->input = layerInit(inputNeuronCount, hiddenLayerNeuronCount[0]);
    nn->hidden = malloc(hiddenLayerCount * sizeof(struct layer*));
    if (!nn->hidden) {
        printf("ERROR: Not enough stack memory! nn->hidden\n");
        assert(0);
    }
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
    assert(nn);
    return nn->input->neuronCount;
}

// required for opaque structure
int outputsNN(struct neuralNetwork *nn) {
    assert(nn);
    return nn->output->neuronCount;
}

// Forward Propagation
void runNN(struct neuralNetwork *nn, double *inputData) {
    assert(nn);
    assert(inputData);

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
            for (int i = 0; i < nn->hidden[l - 1]->neuronCount; i++) {
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

// Fisher-Yates shuffle
void shuffle(double **arr, double **arr2, int len) {
    for (int i = 0; i < len - 1; i++) {
        int index = i + rand() % (len - i);
        double *temp_set = arr[index];
        arr[index] = arr[i];
        arr[i] = temp_set;
        temp_set = arr2[index];
        arr2[index] = arr2[i];
        arr2[i] = temp_set;
    }
}

double squaredError(double output, double expected) {
    return 0.5 * (output - expected) * (output - expected);
}

double error(double output, double expected) {
    // squaredError, other options are available
    return squaredError(output, expected);
}

void trainNN(struct neuralNetwork *nn, const double lr, int epochs, double **trainingInputs, double **trainingOutputs, int trainingSets) {
    assert(nn);
    assert(lr > 0);
    assert(epochs > 0);
    // for (int s = 0; s < trainingSets; s++) {
    //     printf("Inputs: ");
    //     for (int i = 0; i < nn->input->neuronCount; i++) {
    //         printf("%lf ", trainingInputs[s][i]);
    //     }
    //     printf("\nOutputs: ");
    //     for (int i = 0; i < nn->output->neuronCount; i++) {
    //         printf("%lf ", trainingOutputs[s][i]);
    //     }
    //     printf("\n");
    // }

    // Forward Propgation
    for (int e = 0; e < epochs; e++) {
        printf("EPOCH: %d\n", e);   
        // Shuffle training data
        shuffle(trainingInputs, trainingOutputs, trainingSets);
        // Forward Prop and calculate error of output
        double **errors = malloc(sizeof(double*) * trainingSets); // error[training sets][output neurons]
        for (int s = 0; s < trainingSets; s++) {
            runNN(nn, trainingInputs[s]);
            errors[s] = malloc(sizeof(double) * nn->output->neuronCount);
            for (int n = 0; n < nn->output->neuronCount; n++) {
                errors[s][n] = error(nn->output->neurons[n]->output, trainingOutputs[s][n]);
            }
            // Display progress
            printNN(nn);
            printf("Expected Outputs: ");
            for (int n = 0; n < nn->output->neuronCount; n++) {
                printf("%lf ", trainingOutputs[s][n]);
            }
            printf("\n\n");
        }
        // Do back propogation

        for (int s = 0; s < trainingSets; s++) {
            free(errors[s]);
        }
        free(errors);
    }
}

// Weights & Biases tools
void assignWeights(struct neuralNetwork *nn, FILE *file) {
    // Reads first layer bias and input weights
    for (int n = 0; n < nn->hidden[0]->neuronCount; n++) {
        fread(&nn->hidden[0]->neurons[n]->bias, sizeof(double), 1, file); // Read bias
        for (int i = 0; i < nn->input->neuronCount; i++) {
            fread(&nn->input->neurons[i]->weights[n], sizeof(double), 1, file);
        }
    }

    // Reads hidden neuron biases and hidden layer weights up to the last
    for (int l = 1; l < nn->hiddenLayerCount; l++) {
        for (int n = 0; n < nn->hidden[l]->neuronCount; n++) {
            fread(&nn->hidden[l]->neurons[n]->bias, sizeof(double), 1, file);
            for (int i = 0; i < nn->hidden[l - 1]->neuronCount; i++) {
                fread(&nn->hidden[l-1]->neurons[i]->weights[n], sizeof(double), 1, file);
            }
        }
    }

    // output biases and last hidden layer weights
    for (int n = 0; n < nn->output->neuronCount; n++) {
        fread(&nn->output->neurons[n]->bias, sizeof(double), 1, file);
        for (int i = 0; i < nn->hidden[nn->hiddenLayerCount - 1]->neuronCount; i++) {
            fread(&nn->hidden[nn->hiddenLayerCount - 1]->neurons[i]->weights[n], sizeof(double), 1, file);
        }
    }
}

void exportWeights(struct neuralNetwork *nn, FILE *file) {
    fwrite(&nn->input->neuronCount, sizeof(int), 1, file);
    fwrite(&nn->output->neuronCount, sizeof(int), 1, file);
    fwrite(&nn->hiddenLayerCount, sizeof(int), 1, file);
    for(int i = 0; i < nn->hiddenLayerCount; i++) {
        fwrite(&nn->hidden[i]->neuronCount, sizeof(int), 1, file);
    }

    // Writes first layer bias and input weights
    for (int n = 0; n < nn->hidden[0]->neuronCount; n++) {
        fwrite(&nn->hidden[0]->neurons[n]->bias, sizeof(double), 1, file); // Write bias
        for (int i = 0; i < nn->input->neuronCount; i++) {
            fwrite(&nn->input->neurons[i]->weights[n], sizeof(double), 1, file);
        }
    }

    // Writes hidden neuron biases and hidden layer weights up to the last
    for (int l = 1; l < nn->hiddenLayerCount; l++) {
        for (int n = 0; n < nn->hidden[l]->neuronCount; n++) {
            fwrite(&nn->hidden[l]->neurons[n]->bias, sizeof(double), 1, file);
            for (int i = 0; i < nn->hidden[l - 1]->neuronCount; i++) {
                fwrite(&nn->hidden[l-1]->neurons[i]->weights[n], sizeof(double), 1, file); // Go through all past neurons and give their bias to this one
            }
        }
    }

    // output biases and last hidden layer weights
    for (int n = 0; n < nn->output->neuronCount; n++) {
        fwrite(&nn->output->neurons[n]->bias, sizeof(double), 1, file);
        for (int i = 0; i < nn->hidden[nn->hiddenLayerCount - 1]->neuronCount; i++) {
            fwrite(&nn->hidden[nn->hiddenLayerCount - 1]->neurons[i]->weights[n], sizeof(double), 1, file);
        }
    }
}