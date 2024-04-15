#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

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
        return CMD_RUN;
    } else if (strcmp(input, "EXPORT") == 0) {
        return CMD_EXPORT_WEIGHTS;
    } else {
        return INVALID_SYMBOL;
    }
}

// You may use the following comamnds at start:
// NEW <int inputs> <int outputs> <int hidden layer count> <int*: hidden layer nodes>
//      Creates a new neural network
//      Restrictions: Max 20 layers
// LOAD <file containing inputs>
//      Loads a neural network from file
//
// You may use the following commands during use:
// RUN <double* input>
//      Runs the neural network. Recomended to be run post training or assigning weights.
// TRAIN <double: lr> <int: epochs> <file containing inputs> <file containg outputs>
//      Trains the neural network based off of training data.
// EXPORT
//      Exports current weights and biases. Reccomended to be done after training
int main(void) {

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
        strcpy(filepath,"");
        strcat(filepath, "weights/");    
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
        int hiddenLayerNeurons[20];
        fread(&inputCount, sizeof(int), 1, file);
        fread(&outputCount, sizeof(int), 1, file);
        fread(&hiddenLayerCount, sizeof(int), 1, file);
        for(int i = 0; i < hiddenLayerCount; i++) {
            fread(hiddenLayerNeurons + i, sizeof(int), 1, file);
        }
        printf("inputs: %d, hidden layers: %d, and outputs %d, and first hidden neurons %d\n", inputCount, hiddenLayerCount, outputCount, hiddenLayerNeurons[0]);
        struct neuralNetwork *nn = nnInit(inputCount, hiddenLayerCount, hiddenLayerNeurons, outputCount);
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

            break;
        } case CMD_EXPORT_WEIGHTS: {
            char filepath[43];
            strcpy(filepath,"");
            strcat(filepath, "weights/");    
            char stem[30];
            scanf("%s", stem);
            strncat(filepath, stem, 30);
            strcat(filepath, ".bin");
            exportWeights(nn, filepath);
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