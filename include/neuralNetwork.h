// layer.h
#pragma once

struct neuron {
    double *weights;
    double bias;
    double output;
    double delta; // For backpropogation
}

struct layer {
    struct neuron *neurons;
    int num_neurons;
};