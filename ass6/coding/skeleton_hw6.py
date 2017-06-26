#!/usr/bin/env python3
#Filename skeleton_HW6.py
#Author: Christian Knoll, Philipp Gabler
#Edited: 20.6.2017

import numpy as np
import numpy.random as rd
import math

## -------------------------------------------------------
## ---------------- HELPER FUNCTIONS ---------------------
## -------------------------------------------------------

def is_probability_distribution(p):
    """Check if p represents a valid probability distribution."""
    p = np.array(p)
    return np.isclose(p.sum(), 1.0) and np.logical_and(0.0 <= p, p <= 1.0).all()


def sample_discrete_pmf(X, PM, N):
    """Draw N samples for the discrete probability mass function PM that is defined over
    the support X.
    
    X ... Support of the RV -- (S,)
    PM ... Probabilities P(X) -- (S,)
    N ... Number of samples -- scalar
    """

    X, PM = np.asarray(X), np.asarray(PM)

    assert is_probability_distribution(PM)

    y = np.zeros(N, dtype = X.dtype)
    cumulativePM = np.cumsum(PM) # build CDF based on PMF
    offset = rd.uniform(0, 1) * (1 / N) # offset to circumvent numerical issues with cumulativePM
    comb = np.arange(offset, 1 + offset, 1 / N) # new axis with N values in the range ]0,1[

    j = 0

    for i in range(0, N):
        while comb[i] >= cumulativePM[j]: # map the linear distributed values comb according to the CDF
            j += 1
        y[i] = X[j]

    return rd.permutation(y) # permutation of all samples


class HMM:
    def __init__(self, A, B, pi):
        """Represents a Hidden Markov Model with given parameters.

        pi ... Prior/initial probabilities -- (N_s,)
        A ... Transition probabilities -- (N_s, N_s)
        B ... Emission probabilities -- (N_s, N_o)
        """

        A, B, pi = np.asarray(A), np.asarray(B), np.asarray(pi)
        
        assert A.shape[0] == B.shape[0] == pi.shape[0]
        assert is_probability_distribution(pi)
        assert all(is_probability_distribution(p) for p in A)
        assert all(is_probability_distribution(p) for p in B)
        
        self.A = A
        self.B = B
        self.pi = pi
        self.N_s = pi.shape[0]  # number of states
        self.N_o = B.shape[1]   # number of possible observations
        self.improved = False

    def set_improved(self, improved):
        self.improved = improved
        
    def viterbi_discrete(self, X):
        """Viterbi algorithm for an HMM with discrete emission probabilities.
        Returns the optimal state sequence q_opt for a given observation sequence X.

        X ... Observation sequence -- (N,)
        """

        X = np.asarray(X)

        # q_opt is the optimal state sequence; this is a default value
        # q_opt = np.array([0, 0, 1, 2])

        # TODO: implement Viterbi algorithm

        N = X.shape[0]
        q_opt = np.zeros(X.shape[0], dtype=np.int64)
        delta = np.zeros((N, self.N_s))
        psi = np.zeros((N, self.N_s), dtype=np.int64)
        # psi.fill(-1)
        # print(psi)

        A = self.A
        B = self.B


        # init
        if self.improved:
            delta[0, :] = np.log(self.pi) + np.log(self.B[:, X[0]])
        else:
            delta[0, :] = self.pi * self.B[:, X[0]]

        psi[0, :] = 0  # redundant (after init .zeros)
        psi.fill(-1)  # not zero since numpy index starts at 0
        # steps
        for n in range(1, N):
            for j in range(self.N_s):
                p_max = np.finfo('d').min
                p_max_i = -1
                for i in range(self.N_s):
                    if self.improved:
                        p = delta[n - 1, i] + np.log(A[i, j])  # improved with log
                    else:
                        p = delta[n - 1, i] * A[i, j]
                    if p > p_max:
                        p_max = p
                        p_max_i = i
                if self.improved:
                    delta[n, j] = p_max + np.log(B[j, X[n]])  # improved with log
                else:
                    delta[n, j] = p_max * B[j, X[n]]
                psi[n, j] = p_max_i

        # print(delta)
        if self.improved:
            q_opt[N - 1] = np.argmax(delta[N - 1, :])
        else:
            q_opt[N - 1] = np.argmax(delta[N - 1, :])

        delta_a = delta[N - 1, :]
        print("P = {}".format(np.sum(np.e ** delta_a)))
        # backtracking
        p = 0.
        for n in range(N-2, -1, -1):
            q_opt[n] = psi[n+1, q_opt[n+1]]
            p = p + np.e ** delta[n, q_opt[n]]
        # print(p)
        return q_opt

    def sample(self, N):
        """Draw a random state and corresponding observation sequence of length N from the model."""
        # TODO: implement sampling from HMM
        Q = np.zeros(N, dtype=np.int64)
        X = np.zeros(N, dtype=np.int64)
        X_S = np.arange(0, 2, dtype=np.int64)
        Q_S = np.arange(0, 3, dtype=np.int64)

        # start at random combination of state and observation
        X[0] = np.random.randint(2)  # random umbrella
        Q[0] = np.random.randint(3)

        for i in range(1, N):
            # sample next state depending on A distribution (state transition, observation prob)
            # print(self.A[Q[i-1]])
            # TODO remember probability ditribution P_n with Q_n-1
            Q[i] = sample_discrete_pmf(Q_S, self.A[Q[i-1], :], 1)
            # sample next observation depending on state and distribution B (emission prob)
            # print(self.B[Q[i]])
            X[i] = sample_discrete_pmf(X_S, self.B[Q[i]], 1)

        return Q, X


## -------------------------------------------------------
## ------------- START OF  ASSIGNMENT 6 ------------------
## -------------------------------------------------------
def main():
    # define states
    states = ['s', 'r', 'f']    # 3 States: Sun, Rain, Fog
    observations = ['u', 'nu']

    # define HMM 1
    A1 = np.array([[0.8, 0.05, 0.15],
                   [0.2, 0.6, 0.2],
                   [0.2,0.3,0.5]]) # Transition Prob. Matrix
    B1 = np.array([[0.1, 0.9],
                   [0.8, 0.2],
                   [0.3, 0.7]]) # Emission Prob. rows correspond to states, columns to observations
    pi1 = np.array([1/3, 1/3, 1/3]) # Prior
    hmm1 = HMM(A1, B1, pi1)

    # define HMM 2
    A2 = np.array([[0.6, 0.20, 0.20],
                  [0.05, 0.7, 0.25],
                  [0.05, 0.6, 0.35]]) # Transition Prob. Matrix
    B2 = np.array([[0.3, 0.7],
                  [0.95, 0.05],
                  [0.5, 0.5]]) #Emission prob. rows correspond to states, columns to observations
    pi2 = np.array([1/3, 1/3, 1/3]) # Prior
    hmm2 = HMM(A2, B2, pi2)

    # define observation sequences
    X1 = np.array([0, 0, 1, 1, 1, 0])    # 0 = umbrella
    X2 = np.array([0, 0, 1, 1, 1, 0, 0])  # 1 = no umbrella


    # 1.1.) apply Viterbi to find the optimal state sequence and assign the corresponding states
    # TODO: implement in HMM.viterbi_discrete

    # --- example usage of viterbi_discrete:
    hmm2.set_improved(improved=True)
    hmm1.set_improved(improved=True)

    print("X1/HMM1")
    optimal_state_sequence = hmm1.viterbi_discrete(X1)
    
    print(optimal_state_sequence)
    print([states[i] for i in optimal_state_sequence])

    print("X1/HMM2")
    optimal_state_sequence = hmm2.viterbi_discrete(X1)

    print(optimal_state_sequence)
    print([states[i] for i in optimal_state_sequence])

    print("X2/HMM1")
    optimal_state_sequence = hmm1.viterbi_discrete(X2)

    print(optimal_state_sequence)
    print([states[i] for i in optimal_state_sequence])

    print("X2/HMM2")
    optimal_state_sequence = hmm2.viterbi_discrete(X2)

    print(optimal_state_sequence)
    print([states[i] for i in optimal_state_sequence])

    # 1.2.) Sequence Classification
    # TODO
    # done in viterbi and documentation

    # 1.3.) Sample from HMM
    # TODO: implement in HMM.sample
    N = 5
    Q, X = hmm1.sample(N)
    print("Sampling from HMM1")
    print([observations[i] for i in X])
    print([states[i] for i in Q])

    Q, X = hmm2.sample(N)
    print("Sampling from HMM2")
    print([observations[i] for i in X])
    print([states[i] for i in Q])

    # 1.4) Markov chain
    print("Markov-Chain for HMM1")
    A = A1
    pi = pi1
    print("P_1 = ")
    print(pi)
    print("P_2 = ")
    print(np.dot(pi, A ** 1))
    print("P_2 = ")
    print(np.dot(pi, A ** 2))


if __name__ == '__main__':
    main()
