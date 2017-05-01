from sklearn.metrics import confusion_matrix, mean_squared_error

from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.preprocessing import normalize

from nn_classification_plot import plot_hidden_layer_weights, plot_histogram_of_acc, plot_image
import numpy as np



__author__ = 'bellec,subramoney'

"""
Computational Intelligence TU - Graz
Assignment 2: Neural networks
Part 1: Regression with neural networks

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODO!
"""


def ex_2_1(input2, target2):
    '''
    • Write code to train a feed-forward neural network with 1 hidden layers containing 6 hidden units
    for pose recognition. Use dataset2 for training after normalization, ‘adam’ as the training solver and
    train for 200 iterations.
    • Calculate the confusion matrix
    • Plot the weights between each input neuron and the hidden neurons to visualize what the network
    has learnt in the first layer.
    inote Use scikit-learn’s confusion_matrix function to to calculate the confusion matrix. Documentation
    for this can be found here
    inote You can use the coefs_ attribute of the model to read the weights. It is a list of length nlayers − 1
    where the ith element in the list represents the weight matrix corresponding to layer i.
    inote Use the plot_hidden_layer_weights in nn_classification_plot.py to plot the hidden weights.
    '''

    # dataset2 = normalize(input2) already done by main
    x_train = input2
    y_train = target2[:, 1]
    # print(y_train)
    nn = MLPClassifier(solver='adam',
                       activation='tanh',
                       max_iter=200,
                       hidden_layer_sizes=(6,))
    nn.fit(x_train, y_train)
    cm = confusion_matrix(y_train, nn.predict(x_train))
    plot_hidden_layer_weights(nn.coefs_[0])
    print(cm)
    pass


def ex_2_2(input1, target1, input2, target2):
    ## TODO
    pass

