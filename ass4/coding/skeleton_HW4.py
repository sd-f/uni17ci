#Filename: HW4_skeleton.py
#Author: Florian Kaum
#Edited: 15.5.2017
#Edited: 19.5.2017 -- changed evth to HW4

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math
import sys
from scipy.stats import multivariate_normal

def plotGaussContour(mu,cov,xmin,xmax,ymin,ymax,title):
	npts = 100
	delta = 0.025
	stdev = np.sqrt(cov)  # make sure that stdev is positive definite

	x = np.arange(xmin, xmax, delta)
	y = np.arange(ymin, ymax, delta)
	X, Y = np.meshgrid(x, y)

	#matplotlib.mlab.bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0, mux=0.0, muy=0.0, sigmaxy=0.0) -> use cov directly
	Z = mlab.bivariate_normal(X,Y,stdev[0][0],stdev[1][1],mu[0], mu[1], cov[0][1])
	plt.plot([mu[0]],[mu[1]],'r+') # plot the mean as a single point
	CS = plt.contour(X, Y, Z)
	plt.clabel(CS, inline=1, fontsize=10)
	plt.title(title)
	plt.show()
	return

def ecdf(realizations):
	x = np.sort(realizations)
	Fx = np.linspace(0,1,len(realizations))
	return Fx,x


#START OF CI ASSIGNMENT 4
#-----------------------------------------------------------------------------------------------------------------------

# positions of anchors
p_anchor = np.array([[5,5],[-5,5],[-5,-5],[5,-5]])
NrAnchors = np.size(p_anchor,0)

# true position of agent
p_true = np.array([[2,-4]])

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

#1.2) maximum likelihood estimation of models---------------------------------------------------------------------------
#1.2.1) finding the exponential anchor----------------------------------------------------------------------------------
#TODO

#1.2.3) estimating the parameters for all scenarios---------------------------------------------------------------------

#scenario 1
data = np.loadtxt('HW4_1.data',skiprows = 0)
NrSamples = np.size(data,0)
#TODO

#scenario 2
data = np.loadtxt('HW4_2.data',skiprows = 0)
NrSamples = np.size(data,0)
#TODO

#scenario 3
data = np.loadtxt('HW4_3.data',skiprows = 0)
NrSamples = np.size(data,0)
#TODO

#1.3) Least-Squares Estimation of the Position--------------------------------------------------------------------------
#1.3.2) writing the function LeastSquaresGN()...(not here but in this file)---------------------------------------------
#TODO

#1.3.3) evaluating the position estimation for all scenarios------------------------------------------------------------

# choose parameters
#tol = ... # tolerance
#maxIter = ...  # maximum number of iterations

# store all N estimated positions
p_estimated = np.zeros((NrSamples, 2))

for scenario in range(1,5):
	if(scenario == 1):
		data = np.loadtxt('HW4_1.data', skiprows=0)
	elif(scenario == 2):
		data = np.loadtxt('HW4_2.data', skiprows=0)
	elif(scenario == 3):
		data = np.loadtxt('HW4_3.data', skiprows=0)
	elif(scenario == 4):                          
    #scenario 2 without the exponential anchor
		data = np.loadtxt('HW4_2.data', skiprows=0)
	NrSamples = np.size(data, 0)

	#perform estimation---------------------------------------
	# #TODO
	for i in range(0, NrSamples):
		dummy = i

	# calculate error measures and create plots----------------
	#TODO

#1.4) Numerical Maximum-Likelihood Estimation of the Position (scenario 3)----------------------------------------------
#1.4.1) calculating the joint likelihood for the first measurement------------------------------------------------------
#TODO

#1.4.2) ML-Estimator----------------------------------------------------------------------------------------------------

#perform estimation---------------------------------------
#TODO

#calculate error measures and create plots----------------
#TODO

#1.4.3) Bayesian Estimator----------------------------------------------------------------------------------------------

#perform estimation---------------------------------------
#TODO

#calculate error measures and create plots----------------
#TODO


