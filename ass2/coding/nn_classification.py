from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score

from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt

from nn_classification_plot import plot_hidden_layer_weights, plot_histogram_of_acc, plot_image
import numpy as np



__author__ = 'bellec,subramoney, lucas.reeh'

"""
Computational Intelligence TU - Graz
Assignment 2: Neural networks
Part 1: Regression with neural networks

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODO!
"""

IMAGE_DIM = (32, 30)

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
    '''
    • Write code to train a feed-forward neural network with 1 hidden layer containing 20 hidden units
      for recognising the individuals. Use dataset1 for training, ‘adam’ as the training solver and train for
      1000 iterations. Use dataset2 as the test set.
    • Repeat the process 10 times starting from a different initial weight vector and plot the histogram
      for the resulting accuracy on the training and on the test set (the accuracy is proportion of correctly
      classified samples and it is computed with the method score of the classifier).
    • Use the best network (with maximal accuracy on the test set) to calculate the confusion matrix for
      the test set.
    • Plot a few misclassified images.
    '''
    x_train = input1
    y_train = target1[:, 0]
    x_test = input2
    y_test = target2[:, 0]
    seeds = np.array(range(1, 11))
    train_accs = []
    test_accs = []
    max_acc = -1

    for index_seed, seed in np.ndenumerate(seeds):
        nn = MLPClassifier(solver='adam',
                           activation='tanh',
                           max_iter=1000,
                           hidden_layer_sizes=(20,),
                           random_state=seed)
        nn.fit(x_train, y_train)
        train_acc = accuracy_score(y_train, nn.predict(x_train))
        train_accs.append(train_acc)
        test_acc = accuracy_score(y_test, nn.predict(x_test))
        test_accs.append(test_acc)
        if test_acc > max_acc:
            max_acc = test_acc
            best_nn = nn

    plot_histogram_of_acc(train_accs, test_accs)

    cm = confusion_matrix(y_test, best_nn.predict(x_test))
    prediction = best_nn.predict(x_test)
    misclassified = np.where(y_test != prediction)
    print(cm)
    limit = 8
    i = 0
    for mc_index in misclassified[0]:
        if i < limit:
            fig, plts = plt.subplots(1, 2)
            plts[0].set_title("Predicted Person " + str(prediction[mc_index]))
            plts[0].imshow(input2[prediction[mc_index]].reshape(*IMAGE_DIM).T, cmap=plt.cm.gray)
            plts[0].set_xticks(())
            plts[0].set_yticks(())
            plts[1].set_title("Should be Person " + str(y_test[mc_index]))
            plts[1].imshow(input2[y_test[mc_index]].reshape(*IMAGE_DIM).T, cmap=plt.cm.gray)
            plts[1].set_xticks(())
            plts[1].set_yticks(())
            plt.show()
        i = i + 1

    pass


# def get_image(class_id, test_data):
# not working from lecture
'''
def plot_cm(matrix):
    for hidden_neuron_num in range(matrix.shape[1])[:10]:
        plt.figure()
        vmin, vmax = matrix.min(), matrix.max()
        plt.imshow(matrix[:, hidden_neuron_num].reshape(*IMAGE_DIM).T, cmap='Greys',
                   vmin=.5 * vmin, vmax=.5 * vmax, interpolation='none')
        plt.xticks(())
        plt.yticks(())
    plt.close()
    plt.show()

    pass
'''