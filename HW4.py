# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 17:05:12 2018

@author: Shane
"""

import math
import scipy.integrate as integrate
import random
import numpy as np
import sympy as sym

'''
Question 1

plug and play into formula for VC generalization
'''


print('Question 1')
error = 0.05

for N in [400000,420000,440000,460000,480000]:
    print(4* (2*N)**10 * math.exp(-(1/8)*error**2 * N))


"""
Question 2

evaluate the formulae with given quantities, then compare
mH(N) is bounded by dVC, so we substitute
""" 

print('Question 2')

d=50
sigma=.05
N=10000

print(math.sqrt((8/N)*math.log((4*(2*N)**d) /sigma)))
print(math.sqrt((2/N)*math.log((2*N*(N)**d))) + math.sqrt((2/N)*math.log(1/sigma)) + 1/N)
e = sym.Symbol('e')
f = sym.Eq(sym.sqrt((1/N) * (2*e + sym.log(6/sigma * (2*N)**d))),e)
print(sym.solve(f, e))

#bound4 cannot calculate with exact fidelity very easily, so we get an approximation
test_range = 1000000
for i in range(test_range):
    e = i/test_range

    if e < math.sqrt(1/(2*N) * (4*e*(1+e) + math.log(4/sigma) + d*math.log(N**2))):
        continue
    else:
        print(e)
        break




"""
Question 3

same as question 2 but with different N
however, since mH(N) is not bounded by dVC, we cannot substitute and instead the growuth function should be 2**N
"""

print('Question 3')

d=50
sigma=.05
N=5


print(math.sqrt((8/N)*math.log((4*2**(2*N)) /sigma)))
print(math.sqrt((2/N)*math.log((2*N*2**N)) + math.sqrt((2/N)*math.log(1/sigma)) + 1/N))
e = sym.Symbol('e')
f = sym.Eq(sym.sqrt((1/N) * (2*e + sym.log(6/sigma * 2**(2*N)))),e)
print(sym.solve(f, e))

#bound4 cannot calculate with exact fidelity very easily, so we get an approximation
test_range = 1000000
for i in range(test_range):
    e = i/test_range

    if e < math.sqrt(1/(2*N) * (4*e*(1+e) + math.log(4/sigma) + math.log(2**N**2))):
        continue
    else:
        print(e)
        break




"""
Question 4

For a given sine curve, we can select 2 points (c,d) between -1 and 1. It seems like the solution is to evaluate it experimentally rather than mathematically, so will run it over a number of iterations

"""

print('Question 4')
nruns = 1000
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

"""
Question 6

To calculate the variance, can evaluate experimentally using the a that we have generated

"""

print('Question 6')

variance = []
for slope in a:

    #evaluate variance on one point set of data
    h = lambda x: ((slope*x - a_bar*x)**2)/2
    variance.append(integrate.quad(h, -1, 1)[0])

print(np.average(variance))

"""
Question 7

We can do this empirically again, as evaluating it explicitly for two points is quite challenging to get the correct expectations

"""

print('Question 7')

errors = []
for run in range(nruns):
    c = random.uniform(-1,1)
    d = random.uniform(-1,1)

    #solve for best slope using different equations
    y = np.array([math.sin(math.pi*c), math.sin(math.pi*d)])
    X1 = np.matrix([1, 1])
    X2 = np.matrix([c, d])
    X3 = np.matrix([[1, c], [1, d]])
    X4 = np.matrix([c**2, d**2])
    X5 = np.matrix([[1, c**2], [1, d**2]])

    #calculate betas
    beta1 = y * X1.T * np.linalg.inv(X1*X1.T)
    beta2 = y * X2.T * np.linalg.inv(X2*X2.T)
    beta3 = y * X3.T * np.linalg.inv(X3*X3.T)
    beta4 = y * X4.T * np.linalg.inv(X4*X4.T)
    beta5 = y * X5.T * np.linalg.inv(X5*X5.T)


    #evaluate out of sample error for each hypothesis
    f1 = lambda x: ((beta1[0] - math.sin(math.pi*x))**2)/2
    f2 = lambda x: ((beta2[0]*x - math.sin(math.pi*x))**2)/2
    f3 = lambda x: ((beta3[0,0] + beta3[0,1]*x - math.sin(math.pi*x))**2)/2
    f4 = lambda x: ((beta4*x**2 - math.sin(math.pi*x))**2)/2
    f5 = lambda x: ((beta5[0,0] + beta5[0,1]*x**2 - math.sin(math.pi*x))**2)/2

    error = []
    error.append(integrate.quad(f1, -1, 1)[0])
    error.append(integrate.quad(f2, -1, 1)[0])
    error.append(integrate.quad(f3, -1, 1)[0])
    error.append(integrate.quad(f4, -1, 1)[0])
    error.append(integrate.quad(f5, -1, 1)[0])


errors.append(error)

print(np.average(errors, 0))


"""
Question 8

We solve explicitly for Dvc by showing when mH(N) is no loinger equal to 2^N

"""

print('Question 8')

print('Observe that as long as q > N, the equality holds as the combinatorial term goes to 0')
print('Hence, once N = q, then the equality fails. This occurs for mH(N+1)')
print('The largest N where the equality holds is therefore q')

"""
Question 9

First, we evaluate which statements are true. 
a) is true because the intersection of all the sets cannot be larger than the size of any given set
b) is true for the same reason, and therefore is tighter than a
c) is also true for this reason, but is looser than b
d) is false because we already established that the min is a tight upper bound
e) is false for the same reason

"""

print('Question 9')
print('b')

"""
Question 10

First, we evaluate which statements are true. 
a) is false, because assuming the sets were completely distinct, the additional constant term in the VC dimension calcuation would cause the sum of the separate VCs to exceed the VC of the union (for example, the perceptron)
b) is true if we can assume that there is only ever a single constant term added in the VC analysis (not rigourously shown, but it is a result from the slides)
c) is false for the same reason as a
d) is false for the same reason as a
e) is true and tighter as the union must at least be the size of the largest set, and the VC dimension would follow accordingly

"""

print('Question 10')
print('e')
