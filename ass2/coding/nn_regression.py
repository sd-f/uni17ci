import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
import matplotlib.pyplot as plt

from nn_regression_plot import plot_mse_vs_neurons, plot_mse_vs_iterations, plot_learned_function, \
    plot_mse_vs_alpha,plot_bars_early_stopping_mse_comparison

"""
Computational Intelligence TU - Graz
Assignment 2: Neural networks
Part 1: Regression with neural networks

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODO!
"""

__author__ = 'bellec,subramoney'


def calculate_mse(nn, x, y):
    """
    Calculate the mean squared error on the training and test data given the NN model used.
    :param nn: An instance of MLPRegressor or MLPClassifier that has already been trained using fit
    :param x: The data
    :param y: The targets
    :return: Training MSE, Testing MSE
    """

    mse = mean_squared_error(y, nn.predict(x))
    return mse


def ex_1_1_a(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 a)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    params_n_h = [2, 8, 40, 100]
    for n_h in params_n_h:
        nn = MLPRegressor(solver='lbfgs',
                          max_iter=200,
                          activation='logistic',
                          hidden_layer_sizes=(n_h, ),
                          alpha=0, verbose=False,
                          random_state=0)
        # zero randomness
        # verbose=True
        nn.fit(x_train, y_train)
        y_pred_train = nn.predict(x_train)
        y_pred_test = nn.predict(x_test)
        plot_learned_function(n_h, x_train, y_train, y_pred_train, x_test, y_test, y_pred_test)

    pass

def ex_1_1_b(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 b)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    seeds = range(1, 11)
    train_mses = []
    test_mses = []
    for seed in seeds:
        nn = MLPRegressor(solver='lbfgs',
                          max_iter=200,
                          activation='logistic',
                          hidden_layer_sizes=(40, ),
                          alpha=0,
                          random_state=seed)
        nn.fit(x_train, y_train)
        train_mses.append(calculate_mse(nn, x_train, y_train))
        test_mses.append(calculate_mse(nn, x_test, y_test))
        print(seed)
    print("Training MSE"
          + " min = " + str(float(np.min(np.array(train_mses))))
          + " max = " + str(float(np.max(np.array(train_mses))))
          + " mean = " + str(float(np.mean(np.array(train_mses))))
          + " std = " + str(float(np.std(np.array(train_mses)))))
    print("Test MSE"
          + " min = " + str(float(np.min(np.array(test_mses))))
          + " max = " + str(float(np.max(np.array(test_mses))))
          + " mean = " + str(float(np.mean(np.array(test_mses))))
          + " std = " + str(float(np.std(np.array(test_mses)))))

    """
    plt.figure(figsize=(10, 7))
    plt.plot(seeds, train_mses, 'b-', label='Training')
    plt.plot(seeds, test_mses, 'r-', label='Test')
    plt.ylabel('MSE')
    plt.xlabel('Seed')
    plt.legend()
    plt.show()
    """

    # plt.errorbar(seeds, np.zeros(np.array(train_mses).shape), np.array(train_mses), fmt='o')
    # plt.show()
    pass


def ex_1_1_c(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 c)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    params_n_h = np.array([1, 2, 3, 4, 6, 8, 12, 20, 40])
    seeds = np.array(range(1, 11))
    train_mses = np.zeros((params_n_h.shape[0], seeds.shape[0]))
    test_mses = np.zeros((params_n_h.shape[0], seeds.shape[0]))
    for index_seed, seed in np.ndenumerate(seeds):
        for index_n_h, n_h in np.ndenumerate(params_n_h):

            nn = MLPRegressor(solver='lbfgs',
                              max_iter=200,
                              activation='logistic',
                              hidden_layer_sizes=(n_h,),
                              alpha=0,
                              random_state=seed)
            nn.fit(x_train, y_train)
            train_mses[index_n_h, index_seed] = calculate_mse(nn, x_train, y_train)
            test_mses[index_n_h, index_seed] = calculate_mse(nn, x_test, y_test)

    print("Min MSE ", np.min(train_mses))
    plot_mse_vs_neurons(train_mses, test_mses, params_n_h)

    pass

def ex_1_1_d(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 b)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    ## TODO
    pass




def ex_1_2_a(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 a)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    ## TODO
    pass


def ex_1_2_b(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 b)
    Remember to set alpha and momentum to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    ## TODO
    pass

def ex_1_2_c(x_train, x_test, y_train, y_test):
    '''
    Solution for exercise 1.2 c)
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    '''
    ## TODO
    pass