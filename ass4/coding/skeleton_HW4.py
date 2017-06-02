# Filename: HW4_skeleton.py
# Author: Florian Kaum
# Edited: 15.5.2017
# Edited: 19.5.2017 -- changed evth to HW4
# Edited: 1.6.2017 -- Lucas Reeh, ass4 solutions

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import math
import sys
from scipy.stats import multivariate_normal
from numpy import linalg as LA

data1 = np.loadtxt('HW4_1.data')
data2 = np.loadtxt('HW4_2.data')
data3 = np.loadtxt('HW4_3.data')

def plotGaussContour(mu, cov, xmin, xmax, ymin, ymax, title):
    npts = 100
    delta = 0.025
    stdev = np.sqrt(cov)  # make sure that stdev is positive definite

    x = np.arange(xmin, xmax, delta)
    y = np.arange(ymin, ymax, delta)
    X, Y = np.meshgrid(x, y)

    # matplotlib.mlab.bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0, mux=0.0, muy=0.0, sigmaxy=0.0) -> use cov directly
    Z = mlab.bivariate_normal(X, Y, stdev[0][0], stdev[1][1], mu[0], mu[1], cov[0][1])
    plt.plot([mu[0]], [mu[1]], 'r+')  # plot the mean as a single point
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title(title)
    plt.show()
    return


def ecdf(realizations):
    x = np.sort(realizations)
    Fx = np.linspace(0, 1, len(realizations))
    return Fx, x


# START OF CI ASSIGNMENT 4
# -----------------------------------------------------------------------------------------------------------------------

# positions of anchors
p_anchor = np.array([[5, 5], [-5, 5], [-5, -5], [5, -5]])
NrAnchors = np.size(p_anchor, 0)

# true position of agent
p_true = np.array([[2, -4]])

# plot anchors and true position
plt.axis([-6, 6, -6, 6])
for i in range(0, NrAnchors):
    plt.plot(p_anchor[i, 0], p_anchor[i, 1], 'bo')
    plt.text(p_anchor[i, 0] + 0.2, p_anchor[i, 1] + 0.2, r'$p_{a,' + str(i) + '}$')
plt.plot(p_true[0, 0], p_true[0, 1], 'r*')
plt.text(p_true[0, 0] + 0.2, p_true[0, 1] + 0.2, r'$p_{true}$')
plt.xlabel("x/m")
plt.ylabel("y/m")
plt.show()

# 1.2) maximum likelihood estimation of models--------------------------------------------------------------------------
# 1.2.1) finding the exponential anchor---------------------------------------------------------------------------------
def ex_1_2_1(data):

    n_a = np.size(data, 1)
    for a_index in range(0, n_a):
        plt.hist(data[:, a_index], bins=100)

    plt.legend(['a0', 'a1', 'a2', 'a3'])
    plt.title('Histograms')

    plt.show()


# 1.2.3) estimating the parameters for all scenarios--------------------------------------------------------------------
def ex_1_2_3():
    # scenario 1
    print('Scenario 1')
    NrSamples = np.size(data1, 0)
    mys = np.zeros(4)
    sigmas = np.zeros(4)
    n_a = np.size(data1, 1)
    for a_index in range(n_a):
        r = data1[:, a_index]
        mys[a_index] = np.sum(r) / NrSamples
        for s_index in range(NrSamples):
            sigmas[a_index] = sigmas[a_index] + (r[s_index] - mys[a_index] ** 2)
        sigmas[a_index] = sigmas[a_index] / NrSamples
    x = np.arange(n_a)
    plt.bar(x, mys)
    plt.xlabel('Anchor')
    plt.title('Scenario 1 - Parameter My')
    plt.xticks(x, ('0', '1', '2', '3'))
    plt.show()

    plt.bar(x, sigmas)
    plt.xlabel('Anchor')
    plt.title('Scenario 1 - Parameter Sigma²')
    plt.xticks(x, ('0', '1', '2', '3'))
    plt.show()

    print('My ')
    print(mys)
    print('Sigma ')
    print(sigmas)

    # scenario 2
    NrSamples = np.size(data2, 0)
    print('Scenario 2')
    mys = np.zeros(4)
    sigmas = np.zeros(4)
    lambas = np.zeros(4)
    means = np.zeros(4)
    # anchor 0 exp
    r = data2[:, a_index]
    means[0] = np.sum(r) / NrSamples
    for s_index in range(NrSamples):
        lambas[0] = lambas[0] + (r[s_index] - means[0])
    lambas[0] = lambas[0] / NrSamples


    # anchor 1-3 gaussian
    for a_index in range(1, n_a):
        r = data2[:, a_index]
        mys[a_index] = np.sum(r) / NrSamples
        for s_index in range(NrSamples):
            sigmas[a_index] = sigmas[a_index] + (r[s_index] - mys[a_index] ** 2)
        sigmas[a_index] = sigmas[a_index] / NrSamples
    print('My ')
    print(mys)
    print('Sigma ')
    print(sigmas)
    print('Lambda ')
    print(lambas)

    # scenario 3
    print('Scenario 3')
    NrSamples = np.size(data3, 0)
    lambas = np.zeros(4)
    means = np.zeros(4)
    n_a = np.size(data3, 1)
    for a_index in range(n_a):
        r = data3[:, a_index]
        means[a_index] = np.sum(r) / NrSamples
        for s_index in range(NrSamples):
            lambas[a_index] = lambas[a_index] + (r[s_index] - means[a_index])
        lambas[a_index] = lambas[a_index] / NrSamples
    plt.bar(x, lambas)
    plt.xlabel('Anchor')
    plt.title('Scenario 3 - Parameter Lambda')
    plt.xticks(x, ('0', '1', '2', '3'))
    plt.show()

    print('Lambda ')
    print(lambas)


def jacobian(p, a):
    D = np.zeros((4, 2))
    index = 0
    for row in p:
        d_x = (row[0] - a[0]) / (np.sqrt((row[0] - a[0]) ** 2 + (row[1] - a[1]) ** 2))
        d_y = (row[1] - a[1]) / (np.sqrt((row[0] - a[0]) ** 2 + (row[1] - a[1]) ** 2))
        D[index] = [d_x, d_y]
        index = index + 1
    return D

# 1.3) Least-Squares Estimation of the Position-------------------------------------------------------------------------
# 1.3.2) writing the function LeastSquaresGN()...(not here but in this file)--------------------------------------------
def LeastSquaresGN(p_anchor, p_start, r, max_iter, tol):
    a = p_start
    # from https://github.com/basil-conto/gauss-newton/
    # and https://stackoverflow.com/questions/29699057/implementation-of-the-gauss-newton-method-from-wikipedia-example
    for iteration in range(max_iter):
        D = jacobian(p_anchor, a)
        D_t = np.transpose(D)
        a_1 = a - np.dot(np.linalg.pinv(D), r)
        d_p = np.sqrt(((a_1[0] - a[0]) ** 2) + ((a_1[1] - a[1]) ** 2))
        if d_p < tol:
            return a_1
        a = a_1
    return a


# 1.3.3) evaluating the position estimation for all scenarios-----------------------------------------------------------
def ex_1_3_3():
    # choose parameters
    tol = 0.5  # tolerance
    maxIter = 2  # maximum number of iterations

    # store all N estimated positions
    NrSamples = np.size(data1, 0)
    p_estimated = np.zeros((NrSamples, 2))

    for scenario in range(1, 5):
        if scenario == 1:
            data = np.loadtxt('HW4_1.data', skiprows=0)
        elif scenario == 2:
            data = np.loadtxt('HW4_2.data', skiprows=0)
        elif scenario == 3:
            data = np.loadtxt('HW4_3.data', skiprows=0)
        elif scenario == 4:
            # scenario 2 without the exponential anchor
            data = np.loadtxt('HW4_2.data', skiprows=0)
        NrSamples = np.size(data, 0)

        # perform estimation---------------------------------------
        for i in range(0, NrSamples):
            p_estimated[i] = LeastSquaresGN(p_anchor, np.random.uniform(-5, 5, 2), data[i], maxIter, tol)

        # calculate error measures and create plots----------------
        cov = np.cov(p_estimated)
        plotGaussContour(np.mean(p_estimated), cov)
# 1.4) Numerical Maximum-Likelihood Estimation of the Position (scenario 3)---------------------------------------------
# 1.4.1) calculating the joint likelihood for the first measurement-----------------------------------------------------
# TODO

# 1.4.2) ML-Estimator---------------------------------------------------------------------------------------------------

# perform estimation---------------------------------------
# TODO

# calculate error measures and create plots----------------
# TODO

# 1.4.3) Bayesian Estimator---------------------------------------------------------------------------------------------

# perform estimation---------------------------------------
# TODO

# calculate error measures and create plots----------------
# TODO


def main():
    # ex_1_2_1(data2)
    # ex_1_2_3()
    ex_1_3_3()


if __name__ == '__main__':
    main()
