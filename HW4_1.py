# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 17:05:12 2018

@author: Shane
"""

import math
import scipy.integrate as integrate
import random
import numpy as np

'''
Question 1
'''

#plug and play into formula for VC generalization


print('Question 1')
error = 0.05

for N in [400000,420000,440000,460000,480000]:
    print(4* (2*N)**10 * math.exp(-(1/8)*error**2 * N))


"""
Question 2
"""

print('Question 2')

d=50
sigma=.05
N=10000

print(math.sqrt((8/N)*math.log((4*(2*N)**d) /sigma)))
print(math.sqrt((2/N)*math.log((2*N*(N)**d))) + math.sqrt((2/N)*math.log(1/sigma)) + 1/N)
print(math.sqrt((1/N) * math.log(6/sigma * (2*N)**d)))
print(math.sqrt(1/(2*N) * (math.log(4/sigma) + math.log((N**2)**d))))


"""
Question 3

same as question 2 but with different N
"""

print('Question 3')

d=50
sigma=.05
N=5

print(math.sqrt((8/N)*math.log((4*(2*N)**d) /sigma)))
print(math.sqrt((2/N)*math.log((2*N*(N)**d))) + math.sqrt((2/N)*math.log(1/sigma)) + 1/N)
print(math.sqrt((1/N) * math.log(6/sigma * (2*N)**d)))
print(math.sqrt(1/(2*N) * (math.log(4/sigma) + math.log((N**2)**d))))

"""
Question 4

For a given sine curve, we can select 2 points (c,d) between -1 and 1. It seems like the solution is to evaluate it experimentally rather than mathematically, so will run it over a number of iterations

"""

print('Question 4')
nruns = 10000
a = []
for run in range(nruns):
    c = random.uniform(-1,1)
    d = random.uniform(-1,1)

    #solve for best slope
    y = np.array([math.sin(math.pi*c), math.sin(math.pi*d)])
    X = np.matrix([c, d])
    betas = y * X.T * np.linalg.inv(X*X.T)
    a.append(betas[[0]])

a_bar = np.average(a)
print(a_bar)

"""
Question 5

To calculate the bias, need to evaluate the sum of squared difference (assuming this cost function) over the domain

"""

print('Question 5')
g = lambda x: ((a_bar*x - math.sin(math.pi*x))**2)/2
print(integrate.quad(g, -1, 1))

