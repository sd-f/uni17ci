#!/usr/bin/env python3
#Filename skeleton_HW5.py
#Author: Christian Knoll, Philipp Gabler
#Edited: 01.6.2017
#Edited: 02.6.2017 -- naming conventions, comments, ...

import numpy as np
import numpy.random as rd
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import sklearn.mixture as mix
import math
from math import pi, exp
from scipy.stats import multivariate_normal
from sklearn.preprocessing import normalize

LABELS = np.array(['a', 'e', 'i', 'o', 'y'])
COLORS = np.array(['red', 'green', 'blue', 'yellow', 'orange', 'magenta', 'purple'])

## -------------------------------------------------------    
## ---------------- HELPER FUNCTIONS ---------------------
## -------------------------------------------------------

def sample_discrete_pmf(X, PM, N):
    """Draw N samples for the discrete probability mass function PM that is defined over 
    the support X.
       
    X ... Support of RV -- np.array([...])
    PM ... P(X) -- np.array([...])
    N ... number of samples -- scalar
    """

    assert np.isclose(np.sum(PM), 1.0)
    assert all(0.0 <= p <= 1.0 for p in PM)
    
    y = np.zeros(N)
    cumulativePM = np.cumsum(PM) # build CDF based on PMF
    offsetRand = np.random.uniform(0, 1) * (1 / N) # offset to circumvent numerical issues with cumulativePM
    comb = np.arange(offsetRand, 1 + offsetRand, 1 / N) # new axis with N values in the range ]0,1[
    
    j = 0
    for i in range(0, N):
        while comb[i] >= cumulativePM[j]: # map the linear distributed values comb according to the CDF
            j += 1	
        y[i] = X[j]
        
    return rd.permutation(y) # permutation of all samples


def plot_gauss_contour(mu, cov, xmin, xmax, ymin, ymax, title):
    """Show contour plot for bivariate Gaussian with given mu and cov in the range specified.

    mu ... mean -- [mu1, mu2]
    cov ... covariance matrix -- [[cov_00, cov_01], [cov_10, cov_11]]
    xmin, xmax, ymin, ymax ... range for plotting
    """
    
    npts = 500
    deltaX = (xmax - xmin) / npts
    deltaY = (ymax - ymin) / npts
    stdev = [0, 0]

    stdev[0] = np.sqrt(cov[0][0])
    stdev[1] = np.sqrt(cov[1][1])
    x = np.arange(xmin, xmax, deltaX)
    y = np.arange(ymin, ymax, deltaY)
    X, Y = np.meshgrid(x, y)

    Z = mlab.bivariate_normal(X, Y, stdev[0], stdev[1], mu[0], mu[1], cov[0][1])
    plt.plot([mu[0]], [mu[1]], 'r+') # plot the mean as a single point
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline = 1, fontsize = 10)
    plt.title(title)
    # plt.show()


def likelihood_bivariate_normal(X, mu, cov):
    """Returns the likelihood of X for bivariate Gaussian specified with mu and cov.

    X  ... vector to be evaluated -- np.array([[x_00, x_01], ..., [x_n0, x_n1]])
    mu ... mean -- [mu1, mu2]
    cov ... covariance matrix -- [[cov_00, cov_01],[cov_10, cov_11]]
    """
    
    dist = multivariate_normal(mu, cov)
    P = dist.pdf(X)
    return P


## -------------------------------------------------------    
## ------------- START OF  ASSIGNMENT 5 ------------------
## -------------------------------------------------------


def EM(X, M, alpha_0, mu_0, Sigma_0, max_iter):
    # 1. init
    alpha = alpha_0
    mu = mu_0
    Sigma = Sigma_0

    N = X.shape[0]
    R = np.zeros((N, M))
    P = np.zeros((M, N))
    N_m = np.zeros(M)
    LP = np.zeros(max_iter)
    eps = 1



    # init lh, will be done by 4. in further iterations
    for m in range(M):
        # pre computation (for all samples)
        P[m] = likelihood_bivariate_normal(X, mu[m, 0], Sigma[m, 0])

    for t in range(max_iter - 1):
        if t % 4 == 0:
            print("{} % (iteration = {})".format(t / max_iter * 100, t))
        # 2. expectation step
        for n in range(N):
            for m in range(M):
                tmpsum = 0
                for m_prime in range(M):
                    tmpsum = tmpsum + alpha[m_prime, t] * P[m_prime, n]
                if tmpsum != 0:
                    R[n, m] = (alpha[m, t] * P[m, n]) / tmpsum

        # scatter for soft-classification
        for m in range(M):
            samples = R[:, m]
            plt.scatter(X[np.where(samples > 0.5), 0], X[np.where(samples > 0.5), 1], s=1, color=COLORS[m])
        plt.title("Soft-classification snapshot in iteration {}".format(t))
        plt.savefig("figures/sc_{}".format(t))
        plt.clf()
        # plt.show()

        # 3. maximisation step
        for m in range(M):
            N_m[m] = R[:, m].sum()
            # new mu
            tmpsum = np.zeros((1, 2))
            for n in range(N):
                tmpsum = tmpsum + R[n, m] * X[n]
            mu[m, t + 1] = 1.0 / N_m[m] * tmpsum
            # new sigma
            tmpsum = np.zeros((2, 2))
            for n in range(N):
                tmpsum = tmpsum + (X[n] - mu[m, t + 1]).reshape((1, 2)) * R[n, m] * (X[n] - mu[m, t + 1]).reshape((1, 2)).T
            Sigma[m, t + 1] = (1.0 / N_m[m] * tmpsum)  # * np.eye(2)  # make sure diagonal
            # new alpha
            alpha[m, t + 1] = N_m[m] / N
        #print(alpha[:, i + 1])
        #print(mu[:, i + 1])
        #print(Sigma[:, i + 1])

        #print(P)
        # print(N_m)
        # 4. likelihood calculation (for all samples)
        for m in range(M):
            P[m] = likelihood_bivariate_normal(X, mu[m, t + 1], Sigma[m, t + 1])
        for n in range(N):
            tmpsum = 0
            for m in range(M):
                tmpsum = tmpsum + alpha[m, t + 1] * P[m, n]
            LP[t] = LP[t] + np.log(tmpsum)
        #print(LP[i])
        if abs(abs(LP[t]) - abs(LP[t-1])) < eps:
            return alpha[:, t + 1], mu[:, t + 1], Sigma[:, t + 1], LP, t, R
    return alpha[:, t + 1], mu[:, t + 1], Sigma[:, t + 1], LP, t, R


def k_means(X, M, mu_0, max_iter):
    # 1. Init
    mu = mu_0
    Sigma = np.eye(2)
    N = X.shape[0]
    r = np.zeros(M)

    J = np.zeros(max_iter)
    eps = 1
    for t in range(max_iter - 1):
        if t % 4 == 0:
            print("{} % (iteration = {})".format(t / max_iter * 100, t))
        Y = []
        # 2. classification
        # argmin m [(X_n - mu_m).T * (X_n - my_m)
        for n in range(N):
            for m in range(M):
                b = (X[n] - mu_0[m, t])
                r[m] = b.T.dot(b)
            k = np.argmin(r)
            Y.append((k, np.array(X[n])))
        # 3. new mu
        for m in range(M):
            Y_k = [y for y in Y if y[0] == m]
            tmpsum = np.zeros((1, 2))
            for y in Y_k:
                tmpsum = tmpsum + y[1]
            mu[m, t + 1] = tmpsum / len(Y_k)

        tmpsum = 0
        for m in range(M):
            Y_k = [y for y in Y if y[0] == m]

            for y in Y_k:
                b = (y[1] - mu_0[m, t + 1])
                tmpsum = tmpsum + b.T.dot(b)
        J[t + 1] = tmpsum
        plot_Y_scatter(X, M, Y, mu[:, t + 1])
        plt.title("K-means classification snapshot in iteration {}".format(t))
        plt.savefig("figures/kmcl_{}".format(t))
        plt.clf()
        if abs(J[t + 1] - J[t]) < eps:
            return mu[:, t + 1], Y, t, J
    return mu[:, t + 1], Y, t, J


def sample_GMM(alpha, mu, Sigma, N):

    M = len(alpha)
    Y = np.empty((N, 2))
    PM = alpha
    sample = np.zeros([M, 2])
    for n in range(N):
        for i in range(M):
            sample[i] = np.random.multivariate_normal(mu[i], Sigma[i])
        # print(sample[:, 0])
        # TODO fix indipendence in I sampling
        # function cannot work with matrix for parameter X
        Y[n][0] = sample_discrete_pmf(sample[:, 0], PM, 1)
        Y[n][1] = sample_discrete_pmf(sample[:, 1], PM, 1)
    # print(Y)
    return Y


def a1(X):
    # 1.) EM algorithm for GMM:
    # guessing initial theta
    M = 5
    max_iter = 100
    X_norm = X  # normalize(X, axis=0, norm='max')  # scale to 0..1
    alpha_0 = np.ones([M, max_iter])
    mu_0 = np.zeros((M, max_iter, 2))
    Sigma_0 = np.zeros((M, max_iter, 2, 2))
    N = X_norm.shape[0]
    mu = (1.0 / N * np.sum(X_norm, axis=0)).reshape((1, 2))

    for n in range(N):
        tmpsum = (X_norm[n] - mu) * (X_norm[n] - mu).T
    sigma = (1.0 / N * tmpsum) * np.eye(2)

    for m in range(M):
        mu_0[m, 0] = X_norm[np.random.randint(0, N)]
        Sigma_0[m, 0, :, :] = sigma
        alpha_0[m, 0] = alpha_0[m, 0] / M

    # print(X_norm)
    # plt.scatter(X_norm[:, 0], X_norm[:, 1], s=1, color='red')

    alpha, mu, Sigma, LP, t, R = EM(X_norm, M, alpha_0, mu_0, Sigma_0, max_iter)

    # plt.show()
    for m in range(M):
        samples = R[:, m]
        plt.scatter(X[np.where(samples > 0.5), 0], X[np.where(samples > 0.5), 1], s=2, color=COLORS[m])
    plt.scatter(X_norm[:, 0], X_norm[:, 1], s=1, color='lightgrey')
    # print(mu[m])
    # print(Sigma[m])
    for m in range(M):
        plot_gauss_contour(mu[m], Sigma[m],
                           np.min(X_norm, axis=0)[0],
                           np.max(X_norm, axis=0)[0],
                           np.min(X_norm, axis=0)[1],
                           np.max(X_norm, axis=0)[1], 'Gaussian')
    plt.show()
    # print(LP[0:i])
    plt.plot(np.array(range(t)), LP[0:t], '-o')
    plt.savefig("figures/1_2")
    plt.show()

    # plotting log-lh without first iteration
    # plt.plot(np.array(range(1, i)), LP[1:i], '-o')
    # plt.show()

    # plotting compares to real

    for m in range(M):
        data = np.loadtxt('data/' + LABELS[m] + '.data', skiprows=0)
        plt.scatter(data[:, 0], data[:, 1], s=1, color=COLORS[m])
        plot_gauss_contour(mu[m], Sigma[m],
                           np.min(X_norm, axis=0)[0],
                           np.max(X_norm, axis=0)[0],
                           np.min(X_norm, axis=0)[1],
                           np.max(X_norm, axis=0)[1], 'Gaussian')

    plt.show()
    pass


def plot_Y_scatter(X, M, Y, mu):
    for m in range(M):
        Y_k = [y for y in Y if y[0] == m]
        px = np.zeros(len(Y_k))
        py = np.zeros(len(Y_k))
        for y in Y_k:
            px = np.append(px, y[1][0])
            py = np.append(py, y[1][1])
        plt.scatter(px, py, s=2, color=COLORS[m])

    # plt.scatter(X[:, 0], X[:, 1], s=1, color='lightgrey')
    for m in range(M):
        plt.scatter(mu[m, 0], mu[m, 1], s=100, color='black', marker='+', lw=4)
        plt.scatter(mu[m, 0], mu[m, 1], s=100, color=COLORS[m], edgecolors='b', marker='+')

def a2(X):
    M = 7
    max_iter = 100
    N = X.shape[0]

    mu_0 = np.zeros((M, max_iter, 2))
    for m in range(M):
        mu_0[m, 0] = X[np.random.randint(0, N)]
    mu, Y, t, J = k_means(X, M, mu_0, max_iter)

    plot_Y_scatter(X, M, Y, mu)
    plt.savefig("figures/2_0")
    plt.show()

    plt.plot(np.array(range(t-1)), J[1:t], '-o')
    plt.savefig("figures/2_0_distance")
    plt.show()


def a3():
    mix.GaussianMixture()
    M = 5
    N = 1000
    alpha = np.empty(M)
    alpha.fill(1/M)

    mu = np.empty((M, 2))
    Sigma = np.empty((M, 2, 2))
    for m in range(M):
        mu[m] = [m + 1, m + 1]
        sigma = np.eye(2)  # * (m + 1)  # np.ones((2, 2))
        # np.fill_diagonal(sigma, (m + 1))
        Sigma[m] = sigma

    Y = sample_GMM(alpha, mu, Sigma, N)
    return Y


def main():
    # load data
    X = np.loadtxt('data/X.data', skiprows = 0) # unlabeled data
    a = np.loadtxt('data/a.data', skiprows = 0) # label: a
    e = np.loadtxt('data/e.data', skiprows = 0) # label: e
    i = np.loadtxt('data/i.data', skiprows = 0) # label: i
    o = np.loadtxt('data/o.data', skiprows = 0) # label: o
    y = np.loadtxt('data/y.data', skiprows = 0) # label: y

    # 1.) EM algorithm for GMM:
    # a1(X)

    # 2.) K-means algorithm:
    a2(X)

    # 3.) Sampling from GMM
    # Y = a3()
    # a2(Y)
    # a1(Y)
    pass


def sanity_checks():
    # likelihood_bivariate_normal
    mu =  [0.0, 0.0]
    cov = [[1, 0.2],[0.2, 0.5]]
    x = np.array([[0.9, 1.2], [0.8, 0.8], [0.1, 1.0]])
    P = likelihood_bivariate_normal(x, mu, cov)
    print(P)

    # plot_gauss_contour(mu, cov, -2.5, 2.5, -2, 2, 'Gaussian')

    # sample_discrete_pmf
    PM = np.array([0.2, 0.5, 0.2, 0.1])
    N = 1000
    X = np.array([1, 2, 3, 4])
    Y = sample_discrete_pmf(X, PM, N)
    
    print('Nr_1:', np.sum(Y == 1),
          'Nr_2:', np.sum(Y == 2),
          'Nr_3:', np.sum(Y == 3),
          'Nr_4:', np.sum(Y == 4))


if __name__ == '__main__':
    # to make experiments replicable (you can change this, if you like)
    rd.seed(23434345)
    
    # sanity_checks()
    main()
    
