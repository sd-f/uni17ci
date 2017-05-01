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

__author__ = 'bellec,subramoney, lucas.reeh'


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

    '''
    • Write code to train a neural network with nh ∈ 2, 8, 20 hidden neurons on one layer and
    calculate the MSE for the testing and training set at each training iteration for a single seed.
    To be able to calculate the MSEs at each iteration, set warm_start to True and max_iter to
    1 when initializing the network. The usage of warm_start always keeps the previously learnt
    parameters instead of reinitializing them randomly when fit is called (see the documentation
    of scikit learn for more information). Then, loop over iterations and successively call the fit
    function and calculate the MSE on both datasets. Use the training solver ‘lbfgs’, for 1000
    iterations. Stack the results in an array with where the first dimension correspond to the
    number of hidden neurons and the second correspond to the number of iterations Use the
    function plot_mse_vs_iterations in nn_regression_plot.py to plot the variation of MSE
    with iterations.
    • Replace the solver by ‘sgd’ or ‘adam’ and compute the MSE across iterations for the same
    values of nh.
    '''

    params_n_h = np.array([2, 8, 20])
    solvers = np.array(['lbfgs', 'sgd', 'adam'])
    seed = np.random.seed(1)
    max_iterations = 1000
    train_mses = np.zeros((params_n_h.shape[0], max_iterations))
    test_mses = np.zeros((params_n_h.shape[0], max_iterations))

    for solver in solvers:
        for index_n_h, n_h in np.ndenumerate(params_n_h):
            nn = MLPRegressor(solver=solver,
                              max_iter=1,
                              warm_start=True,
                              activation='logistic',
                              hidden_layer_sizes=(n_h,),
                              alpha=0,
                              random_state=seed)
            for iteration in range(max_iterations):
                nn.fit(x_train, y_train)
                train_mses[index_n_h, iteration] = calculate_mse(nn, x_train, y_train)
                test_mses[index_n_h, iteration] = calculate_mse(nn, x_test, y_test)
        print("Using solver = " + solver)
        plot_mse_vs_iterations(train_mses, test_mses, max_iterations, params_n_h)
        # plot_mse_vs_iterations(train_mses, test_mses, max_iterations, params_n_h, solver)

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
    '''
    Write code to train a neural network with n = 40 hidden neurons with values of alpha α =
    [10−8, 10−7, 10−6, 10−5, 10−4, 10−3, 10−2, 10−1, 1, 10, 100]. Stack your results in an array where
    the first axis correspond to the regularization parameter and the second to the number of
    random seeds. Use the training solver ‘lbfgs’, for 200 iterations and 10 different random seeds.
    '''
    params_alpha = np.array([10**-8, 10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1, 10, 100])
    seeds = np.array(range(1, 11))
    train_mses = np.zeros((params_alpha.shape[0], seeds.shape[0]))
    test_mses = np.zeros((params_alpha.shape[0], seeds.shape[0]))
    for index_seed, seed in np.ndenumerate(seeds):
        for index_alpha, alpha in np.ndenumerate(params_alpha):
            nn = MLPRegressor(solver='lbfgs',
                              max_iter=200,
                              activation='logistic',
                              hidden_layer_sizes=(40,),
                              alpha=alpha,
                              random_state=seed)
            nn.fit(x_train, y_train)
            train_mses[index_alpha, index_seed] = calculate_mse(nn, x_train, y_train)
            test_mses[index_alpha, index_seed] = calculate_mse(nn, x_test, y_test)
    plot_mse_vs_alpha(train_mses, test_mses, params_alpha)
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
    '''
    • Early stopping requires the definition of a validation set. Split your training set so that half of
    your old training set become your new training set and the rest is your validation set. Watch
    out, it is crucial to permute the order of the training set before splitting because the data in
    given in increasing order of x.
    • Write code to train a neural network with n = 40 and α = 10^−3 on each selection of the
    training set. Train for 2000 iterations using the ‘lbfgs’ solver for 10 different random seeds and
    monitor the error on each set every 20 iterations. For each individual seed, generate the list of
     (1) the test errors after the last iteration
     (2) the test errors when the error is minimal on the validation set
     (3) the ideal test error when it was minimizing the error on the test set.
    '''
    n = 40
    alpha = 10**-3
    seeds = np.array(range(1, 11))
    max_iterations = 2000

    # new training set, every 2nd training sample
    # y_train_shuffled = np.copy(x_train)
    # np.random.shuffle(x_train_shuffled)
    x_train_new = np.array(x_train)[0::2]
    y_train_new = np.array(y_train)[0::2]
    x_validation = np.array(x_train)[1::2]
    y_validation = np.array(y_train)[1::2]
    test_mse_end = np.zeros(seeds.shape[0])
    test_mse_early_stopping = np.zeros(seeds.shape[0])
    test_mse_ideal = np.zeros(seeds.shape[0])
    for index_seed, seed in np.ndenumerate(seeds):
        mse_end = 100.0
        mse_early_stopping = 100.0
        mse_ideal = 100.0
        nn = MLPRegressor(solver='lbfgs',
                          max_iter=1,
                          warm_start=True,
                          activation='logistic',
                          hidden_layer_sizes=(n,),
                          alpha=alpha,
                          random_state=seed)
        for iteration in range(max_iterations):
            nn.fit(x_train_new, y_train_new)
            if iteration % 20 == 0:
                test_mse = calculate_mse(nn, x_test, y_test)
                validation_mse = calculate_mse(nn, x_validation, y_validation)
                if mse_early_stopping > validation_mse:
                    mse_early_stopping = validation_mse
                if mse_ideal > test_mse:
                    mse_ideal = test_mse
            if iteration == (max_iterations-1):
                mse_end = calculate_mse(nn, x_test, y_test)
        test_mse_end[index_seed] = mse_end
        test_mse_early_stopping[index_seed] = mse_early_stopping
        test_mse_ideal[index_seed] = mse_ideal

    plot_bars_early_stopping_mse_comparison(test_mse_end, test_mse_early_stopping, test_mse_ideal)
    pass

def ex_1_2_c(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 c)
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    """
    '''
    Combining the results from all the previous questions, train a network with the ideal number
    of hidden neurons, regularization parameter and solver choice. Use 10 seeds, a validation set
    and early stopping to identify one particular network (a single seed) that performs optimally.
    '''

    n = 8
    alpha = 10 ** -3
    seeds = np.array(range(1, 11))
    solver = 'sgd'
    train_mses = []
    test_mses = []
    for seed in seeds:
        nn = MLPRegressor(solver=solver,
                          max_iter=2000,
                          activation='logistic',
                          hidden_layer_sizes=(n,),
                          alpha=alpha,
                          early_stopping=True,
                          random_state=seed)
        nn.fit(x_train, y_train)
        test_mse = calculate_mse(nn, x_test, y_test)
        train_mses.append(calculate_mse(nn, x_train, y_train))
        test_mses.append(test_mse)
        if np.min(np.array(test_mses)) >= test_mse:
            print("New min MSE: "+str(float(test_mse)) + " seed: " + str(float(seed)))


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

    pass