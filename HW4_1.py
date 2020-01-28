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
We evaluate this between g_bar and the target function f(x)

"""

print('Question 5')
g = lambda x: ((a_bar*x - math.sin(math.pi*x))**2)/2
print(integrate.quad(g, -1, 1))


"""
Question 6

Similarly, we can calculate the variance by calculating the expectation of the sum of squared difference between every realization of the learned function
and the "average" hypothesis g_bar

"""

print('Question 6')

#using the prior array of slopes generated
var = []

for beta in a:
    g = lambda x: ((a_bar*x - beta*x)**2)/2
    var.append(integrate.quad(g, -1, 1))

print(np.average(var))

"""
Question 7

We can analyse the performance of each hypothesis set heuristically. It seems that due to the data complexity, a simpler hypothesis would perform the best. 
This allows for either h(x) = b or h(x) = ax. It further appears that probabilistically, h(x) = ax would fit the curve better that the straight line.

"""

print('Question 7')

print("ax")


"""
Question 8

Given that the definition of the VC dimension is that mH must equal to 2**n, it seems that the relationship holds as long as the permutation term is zero valued.
Hence, mH(N + 1) would no longer satisfy this relationship for q = N

"""

print('Question 8')

print("b")

"""
Question 9

Using general set theory, the intersection cannot be smaller than zero nor larger than the smallest set.

"""

print('Question 9')

print("b")

"""
Question 10

Again, by general set theory the union cannot be smaller than the largest set.
Showing that the union of the hypothesis set can be larger that the sum is a little bit tricky. However, by leveraging what we have already seen in the lectures,
we know that the VC dimension of the 2d perceptron is 3. However, it is clear that the VC dimension of the positive ray (i.e a one way perceptron) is only 1. Ergo, we have a case
where the VC dimension of the sum exceeds the sum of the VC dimension, and it cannot be that d is true.

"""

print('Question 10')

print("e")