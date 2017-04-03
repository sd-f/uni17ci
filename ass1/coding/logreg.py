#!/usr/bin/env python
import numpy as np

from logreg_toolbox import sig

__author__ = 'bellec, subramoney'

"""
Computational Intelligence TU - Graz
Assignment: Linear and Logistic Regression
Section: Gradient descent (GD) (Logistic Regression)
TODO Fill the cost function and the gradient
"""


def cost(theta, x, y):
    """
    Cost of the logistic regression function.

    :param theta: parameter(s)
    :param x: sample(s)
    :param y: target(s)
    :return: cost
    """
    N, n = x.shape

    ##############
    #
    # TODO
    #
    # Write the cost of logistic regression as defined in the lecture

    c = 0
    for i in range(0, N):
        p = sig(x[i].dot(theta))
        if y[i] == 0:
            c += np.log(1 - p)
        else:
            c += np.log(p)

    c = -c / N

    # END TODO
    ###########

    return c


def grad(theta, x, y):
    """

    Compute the gradient of the cost of logistic regression

    :param theta: parameter(s)
    :param x: sample(s)
    :param y: target(s)
    :return: gradient
    """
    N, n = x.shape

    ##############
    #
    # TODO
    #

    g = np.zeros(theta.shape)

    for i in range(0, g.shape[0]):
        sum_ = 0
        for j in range(0, N):
            p = sig(x[j].dot(theta))
            sum_ = sum_ + (p - y[j]) * x[j][i]
        g[i] = sum_ / N

    # END TODO
    ###########

    return g
