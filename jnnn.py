#!/bin/env python
# -*- coding: iso-8859-15 -*-
# (C) 2014 J. Nurminen <slinky@iki.fi>
# 
# This is just for my own neural network investigations, I wanted to finally
# learn backpropagation and play around with a simple implementation where I know
# the location of each single thing.
#
# Basic backpropagation with momentum seems to work OK for AND and XOR functions.
# Though for some reason I had to add one more hidden layer unit than should be
# necessary. Two hidden units should suffice, but I really must have 3.
#
# This isn't any good for "real work" -- use it for learning.

import random
import math

EPOCHS = 1000

# Activation functions
def sigmoid(x):
    return 1.0/(1.0 + math.exp(-x))

def d_sigmoid(x):
    # derivative of sigmoid
    return x*(1.0-x)

def sigmoid_tanh(x):
    return math.tanh(x)

def d_sigmoid_tanh(x):
    # derivative of sigmoid_tanh
    return 1.0 - x**2.0


# Miscellaneous functions
def get_random():
    """ PRNG routine

        Returns:
            Pseudorandom number from the range -1.0 to 1.0
    """
    return random.uniform(-1.0, 1.0)


class Layer:
    """A weighted connection structure between inputs and outputs.

    A network with an input, a hidden and an output layer would consist of two
    Layer objects:
        - input-to-hidden
        - hidden-to-output

    Note that the hidden layer would essentially be shared and not explicitly
    described.
    """
    BIAS_VALUE = -1.0

    def __init__(self, src_size=2, target_size=2, act=None, act_dx=None, has_bias=False, \
            scaler=1.0):
        """Create a connection structure with random weights.

        If act is None, then everything fails horribly.

        Weight layout (ex. 3-->2 layer, src=3, target=2): [[0, 1, 2], [0, 1, 2]]
        Indexed with [target][src].

        Augmented vectors are used, i.e. constant-input bias term is added and
        only the bias weight is varied.

        Args:
            src_size (int): nodes in the source (left) (matrix rows)
            target_size (int): nodes in the target (right) (matrix cols)
            act (func): activation function
            act_dx (func): derivative of activation function
            has_bias (boolean): True if has bias term
            scaler (float): Scaler for random values (0.2 would make PRNG range -0.2 .. 0.2)
        """

        if has_bias:
            src_size += 1
        self.has_bias = has_bias

        # weights are initialized to small random numbers
        self.weights = []
        for j in range(target_size):
            row = [get_random() * scaler for i in range(src_size)]
            self.weights.append(row)

        self.act = act
        self.act_dx = act_dx
        self.out = [0.0] * target_size
        self.target_size = target_size
        self.inputs = [0.0] * src_size
        self.src_size = src_size


    def run(self, inputs):
        """Run a single feedforward iteration.

        The length of inputs must match the src_size given in constructor.

        Returns:
            The output of the iteration
        """
        if self.has_bias:
            # i'm unsure of myself
            assert len(inputs)+1 == self.src_size
        else:
            assert len(inputs) == self.src_size

        for i, val in enumerate(inputs):
            self.inputs[i] = val
            # XXX use deepcopy
        if self.has_bias:
            self.inputs.append(self.BIAS_VALUE)

        for target in range(self.target_size):
            weighted = 0.0
            for src in range(self.src_size):
                weighted += self.weights[target][src] * self.inputs[src]
            self.out[target] = self.act(weighted)
        return self.out


class FFNN:
    """Feed-forward neural network with backpropagation and momentum.
    """
    LAYER_I2H, LAYER_H2O = 0, 1

    def __init__(self, input_size=2, hidden_size=4, output_size=2, act=None, act_dx=None):
        """Takes care of building the network
        """
        self.layers = []

        # input to hidden
        layer = Layer(src_size=input_size, target_size=hidden_size, act=act,
                        act_dx=act_dx, has_bias=True, scaler=0.33)
        self.layers.append(layer)

        # hidden to output
        layer = Layer(src_size=hidden_size, target_size=output_size, act=act,
                        act_dx=act_dx, has_bias=True, scaler=1.5)
        self.layers.append(layer)

        # previous weight deltas for momentum,
        # this basically matches the weight matrices,
        # but is initialized to 0
        self.prev_deltaw = []
        for layer in self.layers:
            self.prev_deltaw.append([[0 for i in range(layer.src_size)] \
                    for j in range(layer.target_size)])

    def run(self, inputs):
        """Run a feedforward iteration for the entire network
        """
        out = inputs
        for i in range(len(self.layers)):
            out = self.layers[i].run(out)
        return out

    def calc_error(self, example):
        """Calculate an error metric over all input-output pairs.

        The metric is mean squared error.

        Returns:
            The error
        """
        error = 0.0
        num_outputs = len(example[0][1])
        for inputs, outputs in example:
            out = self.run(inputs)
            sum_err = 0.0
            for i in range(len(outputs)):
                sum_err += (outputs[i] - out[i])**2
            error += sum_err
        error /= len(example)*num_outputs
        return error

    def train(self, inputs, outputs, learn_rate=0.15, momentum_rate=0.9, verbose=False):
        """Basic backpropagation training

        Currently supports only a 2 layer (input, hidden, output) network!
        """
        if verbose:
            print "Train: %s %s" % (inputs, outputs)

        # first, collect output from all units in the net
        self.run(inputs)

        # calculate output layer errors to backpropagate
        # becomes delta[1][] (to match the layer indexes)
        delta = []
        tmp = []
        layer = self.layers[self.LAYER_H2O]
        for output_value, target_value in zip(layer.out, outputs):
            error = -(target_value - output_value) * layer.act_dx(output_value)
            tmp.append(error)
            if verbose:
                print "OL delta: want=%s out=%s error=%s" % \
                        (target_value, output_value, error)
        delta.insert(0, tmp)

        # calculate hidden layer errors to backpropagate
        # becomes delta[0][]
        layer = self.layers[self.LAYER_H2O]
        tmp = []
        for j in range(layer.src_size):
            error = 0.0
            for k in range(layer.target_size):
                error += delta[self.LAYER_I2H][k] * layer.weights[k][j] * \
                    layer.act_dx(layer.inputs[j])
            tmp.append(error)
            if verbose:
                print "HL delta: error=%s" % (error)
        delta.insert(0, tmp)

        # update weights using the delta values:
        # 1: output layer (hidden to output weights), then
        # 0: hidden layer (input to hidden weights)
        #
        # includes momentum
        for layer_num in reversed(xrange(len(self.layers))):
            layer = self.layers[layer_num]
            for target in range(layer.target_size):
                for src in range(layer.src_size):
                    delta_w = -learn_rate * delta[layer_num][target] * \
                            layer.inputs[src]
                    layer.weights[target][src] += delta_w + momentum_rate * \
                            self.prev_deltaw[layer_num][target][src]
                    self.prev_deltaw[layer_num][target][src] = delta_w

        if verbose:
            print "Weights:"
            print "HL: %s" % (self.layers[self.LAYER_I2H].weights)
            print "OL: %s" % (self.layers[self.LAYER_H2O].weights)


# examples to learn
examples = {
        #name: [input0, ..., inputN-1], [result]
        'AND':[
            [[0, 0], [0.0]],
            [[0, 1], [0.0]],
            [[1, 0], [0.0]],
            [[1, 1], [1.0]],
            ],

        'XOR':[
            [[0, 0], [0.0]],
            [[0, 1], [1.0]],
            [[1, 0], [1.0]],
            [[1, 1], [0.0]]],
}

def learn_online(nn, this_case):
    """Stochastic/online learning, weights are updated after each training
    example.
    """
    iters = 0
    for i in range(EPOCHS):
        inputs, outputs = random.choice(examples[this_case])
        nn.train(inputs, outputs,
                learn_rate=0.15, momentum_rate=0.9,
                verbose=False)
        if i % 50 == 0:
            print "%s: MSE = %s" % (i, nn.calc_error(examples[this_case]))


if __name__ == '__main__':
    random.seed(31337)

    nn = FFNN(input_size=2, hidden_size=3, output_size=1,
            act=sigmoid_tanh,
            act_dx=d_sigmoid_tanh)

    learn_this = 'XOR' # or 'AND'
    learn_online(nn, learn_this)

    print ""
    print "The network:"

    for i, la in enumerate(nn.layers):
        print "Layer %s: %s to %s, %s" % (i, la.src_size, la.target_size, \
                la.weights)

    print ""
    print "The network learnt this:"

    for inputs, outputs in examples[learn_this]:
        print "[%s, %s] --> %s (vs. %s)" % (inputs[0], inputs[1], \
                nn.run(inputs), outputs)

