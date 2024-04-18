#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <time.h>

#include "neuralNetwork.h"

#define INVALID_SYMBOL -1
#define CMD_NEW 10
#define CMD_LOAD 11
#define CMD_RUN 1
#define CMD_TRAIN 2
#define CMD_EXPORT_WEIGHTS 3

int readCommand() {
    char input[7]; // 7 to accomidate size of export word
    if (scanf("%s", input) == EOF) {
        return INVALID_SYMBOL;
    } else if (strcmp(input, "NEW") == 0) {
        return CMD_NEW;
    } else if (strcmp(input, "LOAD") == 0) {
        return CMD_LOAD;
    } else if (strcmp(input, "RUN") == 0) {
        return CMD_RUN;
    } else if (strcmp(input, "TRAIN") == 0) {
        return CMD_TRAIN;
    } else if (strcmp(input, "EXPORT") == 0) {
        return CMD_EXPORT_WEIGHTS;
    } else {
        return INVALID_SYMBOL;
    }
}

// You may use the following comamnds at start:
// NEW <int inputs> <int outputs> <int hidden layer count> <int*: hidden layer nodes>
//      Creates a new neural network
// LOAD <file stem>
//      Loads a neural network weights and biases from file
//
// You may use the following commands during use:
// RUN <double* input>
//      Runs the neural network (Forward propogation). Recomended to be run post training or assigning weights.
// TRAIN <double: lr> <int: epochs> <file containing training data>
//      Trains the neural network based off of training data.
// EXPORT <file stem>
//      Exports current weights and biases. Reccomended to be done after training
int main(void) {
    
    srand(time(NULL));

    //Initialization of neural network
    int inputCount = 0;
    int outputCount = 0;
    int hiddenLayerCount = 0;

    int cmd = INVALID_SYMBOL;
    cmd = readCommand();
    struct neuralNetwork *nn = NULL;

    // Load or create nn
    switch (cmd) {
    case CMD_NEW: {
        scanf("%d", &cmd); // Get input layer count
        inputCount = cmd;
        scanf("%d", &cmd); // Get output layer count;
        outputCount = cmd;
        scanf("%d", &cmd); // Get hidden layer count
        hiddenLayerCount = cmd;
        int *hiddenLayerNeuronCount = malloc(hiddenLayerCount * sizeof(int));
        for (int hl = 0; hl < hiddenLayerCount; hl++) {
            scanf("%d", &cmd);
            hiddenLayerNeuronCount[hl] = cmd;
        }
        nn = nnInit(inputCount, hiddenLayerCount, hiddenLayerNeuronCount, outputCount);
        free(hiddenLayerNeuronCount);
        break;
    } case CMD_LOAD: {
        char filepath[43];
        strcpy(filepath, "weights/");    
        char stem[30];
        scanf("%29s", stem);
        strncat(filepath, stem, 30);
        strcat(filepath, ".bin");
        FILE *file;
        file = fopen(filepath, "rb");
        if (file == NULL) {
            printf("Invalid file. Cannot read %s\n", filepath);
            return 0;
        }
        fread(&inputCount, sizeof(int), 1, file);
        fread(&outputCount, sizeof(int), 1, file);
        fread(&hiddenLayerCount, sizeof(int), 1, file);
        int *hiddenLayerNeurons = malloc(hiddenLayerCount * sizeof(int));
        for(int i = 0; i < hiddenLayerCount; i++) {
            fread(hiddenLayerNeurons + i, sizeof(int), 1, file);
        }
        nn = nnInit(inputCount, hiddenLayerCount, hiddenLayerNeurons, outputCount);
        free(hiddenLayerNeurons);
        assignWeights(nn, file);
        fclose(file);
        break;
    } default: {
        return 0;
        break;
    } }
    // Normal Operation
    cmd = readCommand();
    while (cmd != INVALID_SYMBOL) {
        switch (cmd) {
        case CMD_RUN: {
            int inputCount = inputsNN(nn);
            double *inputs = malloc(inputCount * sizeof(double));
            for (int i = 0; i < inputCount; i++) {
                scanf("%lf", inputs + i);
            }
            runNN(nn, inputs);
            free(inputs);
            printNN(nn);
            break;
        } case CMD_TRAIN: {
            // Training data
            double lr;
            scanf("%lf", &lr);
            int epochs;
            scanf("%d", &epochs);
            // File buisness
            char filepath[43];
            strcpy(filepath, "training/");    
            char stem[30];
            scanf("%s", stem);
            strncat(filepath, stem, 30);
            strcat(filepath, ".bin");
            FILE *file;
            file = fopen(filepath, "rb");
            if (file == NULL) {
                printf("Invalid file. Cannot read %s\n", filepath);
                return 0;
            }
            // File data
            int inputNodes = inputsNN(nn);
            int outputNodes = outputsNN(nn);
            double *in = malloc(sizeof(double) * inputNodes);
            double *out = malloc(sizeof(double) * outputNodes);
            int trainingSets = 0;
            int maxSets = 4;
            double **testsIn = malloc(maxSets * sizeof(double*));
            double **testsOut = malloc(maxSets * sizeof(double*));
            // Read tests from file
            while(!feof(file)) {
                if (fread(in, sizeof(double), inputNodes, file) < inputNodes) {
                    break;
                }
                fread(out, sizeof(double), outputNodes, file);
                if (trainingSets + 1 > maxSets) { // Doubling strategy to allow infinite test cases
                    maxSets *= 2;
                    testsIn = realloc(testsIn, maxSets * sizeof(double*));
                    testsOut = realloc(testsIn, maxSets * sizeof(double*));
                }
                testsIn[trainingSets] = malloc(sizeof(double) * inputNodes);
                testsOut[trainingSets] = malloc(sizeof(double) * outputNodes);
                memcpy(testsIn[trainingSets], in, sizeof(double) * inputNodes);
                memcpy(testsOut[trainingSets], out, sizeof(double) * outputNodes);
                trainingSets++;
            }
            free(in);
            free(out);
            testsIn = realloc(testsIn, trainingSets * sizeof(double*)); // Readjust to correct size
            testsOut = realloc(testsOut, trainingSets * sizeof(double*)); 
            // Train NN
            
            trainNN(nn, lr, epochs, testsIn, testsOut, trainingSets);
            // Clean up
            for (int i = 0; i < trainingSets; i++) { // Free 'in's and 'out's
                free(testsIn[i]);
                free(testsOut[i]);
            }
            free(testsIn);
            free(testsOut);
            break;
        } case CMD_EXPORT_WEIGHTS: {
            char filepath[43];
            strcpy(filepath, "weights/");    
            char stem[30];
            scanf("%s", stem);
            strncat(filepath, stem, 30);
            strcat(filepath, ".bin");
            FILE *file;
            file = fopen(filepath, "wb");
            if (file == NULL) {
                printf("Invalid file. Cannot read %s\n", filepath);
                return 0;
            }
            exportWeights(nn, file);
            fclose(file);
            break;
        } default: {
            return 1;
            break;
        } }
        cmd = readCommand();
    }
    nnFree(nn);
    return 0;
}