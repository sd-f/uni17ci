import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix

from svm_plot import plot_svm_decision_boundary, plot_score_vs_degree, plot_score_vs_gamma, plot_mnist, \
    plot_confusion_matrix

"""
Computational Intelligence TU - Graz
Assignment 3: Support Vector Machine, Kernels & Multiclass classification
Part 1: SVM, Kernels

TODOS are all contained here.
"""

__author__ = 'bellec,subramoney'


def ex_1_a(x, y):
    """
    Solution for exercise 1 a)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Train an SVM with a linear kernel

    clf = svm.SVC(kernel="linear")
    clf.fit(x, y)

    ## and plot the decision boundary and support vectors using 'plot_svm_decision_boundary' function

    plot_svm_decision_boundary(clf, x, y)

    ###########


def ex_1_b(x, y):
    """
    Solution for exercise 1 b)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Add a point (4,0) with label 1 to the data set and then

    x_new = np.vstack((x, (4, 0)))
    y_new = np.hstack((y, 1))

    ## train an SVM with a linear kernel

    ## c = 0.1 --> misclassifaction
    ## c > 2.7 --> works and (4,0) is inside margin
    clf = svm.SVC(kernel="linear")
    clf.fit(x_new, y_new)

    ## and plot the decision boundary and support vectors using 'plot_svm_decision_boundary' function

    plot_svm_decision_boundary(clf, x_new, y_new)

    ###########


def ex_1_c(x, y):
    """
    Solution for exercise 1 c)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Add a point (4,0) with label 1 to the data set and then

    x_new = np.vstack((x, (4, 0)))
    y_new = np.hstack((y, 1))

    ## train an SVM with a linear kernel with different values of C
    ## and plot the decision boundary and support vectors  for each using 'plot_svm_decision_boundary' function
    ###########
    Cs = [1e6, 1, 0.1, 0.001]

    for C in Cs:
        clf = svm.SVC(kernel="linear", C=C)
        clf.fit(x_new, y_new)
        print("Support Vectors (C=" + str(C) + "): " + str(len(clf.support_vectors_)))
        plot_svm_decision_boundary(clf, x_new, y_new)


def ex_2_a(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 a)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train an SVM with a linear kernel for the given dataset

    clf = svm.SVC(kernel="linear")
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print("Score (linear): " + str(score))

    ## and plot the decision boundary and support vectors  for each using 'plot_svm_decision_boundary' function

    plot_svm_decision_boundary(clf, x_train, y_train, x_test, y_test)

    ###########


def ex_2_b(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 b)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train SVMs with polynomial kernels for different values of the degree
    ## (Remember to set the 'coef0' parameter to 1)
    ## and plot the variation of the test and training scores with polynomial degree using 'plot_score_vs_degree' func.
    ## Plot the decision boundary and support vectors for the best value of degree
    ## using 'plot_svm_decision_boundary' function
    ###########
    degrees = np.array(range(1, 20))
    best_score = -1
    train_scores = np.zeros(degrees.shape[0])
    test_scores = np.zeros(degrees.shape[0])
    for i, d in np.ndenumerate(degrees):
        clf = svm.SVC(kernel="poly", degree=d, coef0=1)
        clf.fit(x_train, y_train)
        train_scores[i] = clf.score(x_train, y_train)
        score = clf.score(x_test, y_test)
        test_scores[i] = score
        # print("Score (degree: " + str(d) + "): " + str(score))
        if score > best_score:
            best_score = score
            best_clf = clf

    print("Best score is " + str(np.max(test_scores)) + " at degree = " + str(degrees[np.argmax(test_scores)]))
    plot_score_vs_degree(train_scores, test_scores, degrees)
    plot_svm_decision_boundary(best_clf, x_train, y_train, x_test, y_test)


def ex_2_c(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 c)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train SVMs with RBF kernels for different values of the gamma
    ## and plot the variation of the test and training scores with gamma using 'plot_score_vs_gamma' function.
    ## Plot the decision boundary and support vectors for the best value of gamma
    ## using 'plot_svm_decision_boundary' function
    ###########
    gammas = np.arange(0.01, 2, 0.02)
    best_score = -1
    train_scores = np.zeros(gammas.shape[0])
    test_scores = np.zeros(gammas.shape[0])
    for i, gamma in np.ndenumerate(gammas):
        clf = svm.SVC(kernel="rbf", gamma=gamma)
        clf.fit(x_train, y_train)
        train_scores[i] = clf.score(x_train, y_train)
        score = clf.score(x_test, y_test)
        test_scores[i] = score
        # print("Score (degree: " + str(d) + "): " + str(score))
        if score > best_score:
            best_score = score
            best_clf = clf

    print("Best score is " + str(np.max(test_scores)) + " at gamma = " + str(gammas[np.argmax(test_scores)]))

    plot_score_vs_gamma(train_scores, test_scores, gammas)
    plot_svm_decision_boundary(best_clf, x_train, y_train, x_test, y_test)


def ex_3_a(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 3 a)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train multi-class SVMs with one-versus-rest strategy with
    ## - linear kernel
    ## - rbf kernel with gamma going from 10**-5 to 10**-3
    ## - plot the scores with varying gamma using the function plot_score_versus_gamma
    ## - Mind that the chance level is not .5 anymore and add the score obtained with the linear kernel as optional argument of this function
    ###########

    clf = svm.SVC(kernel="linear", C=3e-4, decision_function_shape='ovr')
    clf.fit(x_train, y_train)
    lin_score_train = clf.score(x_train, y_train)
    lin_score_test = clf.score(x_test, y_test)
    print("Score (linear): " + str(lin_score_test))
    gamma_list = np.arange(10e-5, 10e-3, (10e-3 - 10e-5) / 10.0)
    train_scores = np.zeros(gamma_list.shape[0])
    test_scores = np.zeros(gamma_list.shape[0])
    for i, gamma in np.ndenumerate(gamma_list):
        clf = svm.SVC(kernel="rbf", gamma=gamma, decision_function_shape='ovr', C=3e-4)
        clf.fit(x_train, y_train)
        train_scores[i] = clf.score(x_train, y_train)
        score = clf.score(x_test, y_test)
        test_scores[i] = score

    print("Best score is " + str(np.max(test_scores)) + " at gamma = " + str(gamma_list[np.argmax(test_scores)]))
    plot_score_vs_gamma(train_scores, test_scores, gamma_list, lin_score_train, lin_score_test, np.mean(test_scores))

    # plot_mnist()


def ex_3_b(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 3 b)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train multi-class SVMs with a LINEAR kernel
    ## Use the sklearn.metrics.confusion_matrix to plot the confusion matrix.
    ## Find the index for which you get the highest error rate.
    ## Plot the confusion matrix with plot_confusion_matrix.
    ## Plot the first 10 occurrences of the most misclassified digit using plot_mnist.
    ###########

    labels = range(1, 6)

    clf = svm.SVC(kernel="linear", C=3e-4, decision_function_shape='ovr')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_pred, labels)
    plot_confusion_matrix(cm, labels)
    sel_err = np.where(y_test != y_pred)  # Numpy indices to select images that are misclassified.
    np.fill_diagonal(cm, 0)
    i, j = np.unravel_index(cm.argmax(), cm.shape)
    # Plot with mnist plot
    plot_mnist(x_test[sel_err], y_pred[sel_err], labels=labels[i], k_plots=10, prefix='predicted class')
