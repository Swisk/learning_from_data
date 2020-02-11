# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 17:05:12 2018

@author: Shane
"""

import math
import scipy.integrate as integrate
import random
import numpy as np
from sympy import symbols, diff

'''
Question 1
'''

#plug and play into formula for error


print('Question 1')
sigma = 0.1
d = 8

for N in [1000, 500, 100, 25, 10]:
    if sigma**2*(1 - (d+1)/N) > 0.008:
        print(N)
        break


"""
Question 2
For sufficiently large values of x1**2, the decision is negative. This implies that weight 1 is a negative weight.
In contrast, the decision is positive for very large x2**2, implying a positive weight.

"""

print('Question 2')
print('d')



"""
Question 3

The intuition is that the VC dimension is the number of free parameters. After transformation, we have more free parameters, so it is the case that the VC dimension can be as large as 15.
"""

print('Question 3')
print('d')

"""
Question 4
We can calcuate the partial derivative mathematically.

"""

print('Question 4')
print('e')

"""
Question 5

We need to iterate over the error surface. This can be done by taking the partial derivative and moving a step for both u and v.

"""

print('Question 5')
eta = 0.1
u, v = 1, 1


for i in range(17):
    if float((u*math.exp(v) - 2*v*math.exp(-u))**2) < 10**-14:
        print("{} iterations".format(i))
        break

    u_diff = 2*(math.exp(v) + 2*v*math.exp(-u)) * (u*math.exp(v) - 2*v*math.exp(-u))
    v_diff = 2*(u*math.exp(v) - 2*math.exp(-u)) * (u*math.exp(v) - 2*v*math.exp(-u))

    u = u - eta * u_diff
    v = v - eta * v_diff
    


"""
Question 6
We can just print the final values after the gradient descent implemented earlier

"""

print('Question 6')
print((u, v))

"""
Question 7
Using the same idea, we update alternatingly and then reevaluate before updating the other coordinate.

"""

print('Question 7')
eta = 0.1
u, v = 1, 1


for i in range(14):

    #update alternatingly instead
    u_diff = 2*(math.exp(v) + 2*v*math.exp(-u)) * (u*math.exp(v) - 2*v*math.exp(-u))
    u = u - eta * u_diff
    
    v_diff = 2*(u*math.exp(v) - 2*math.exp(-u)) * (u*math.exp(v) - 2*v*math.exp(-u))
    v = v - eta * v_diff
    
print((u*math.exp(v) - 2*v*math.exp(-u))**2)

"""
Question 8 and 9



"""

N = 100
runs = 100
eta = 0.01

def setup_problem(N):
    points = []
    for i in range(N):
        coord = [random.uniform(-1,1), random.uniform(-1,1)]
        points.append(coord)
        
    points = np.array(points)
    
    # get bisecting line
    point1 = (random.uniform(-1,1), random.uniform(-1,1))
    point2 = (random.uniform(-1,1), random.uniform(-1,1))
    a = (point1[1] - point2[1])/(point1[0] - point2[0])
    b = point1[1] - a*point1[0]
    
    labels = []
    for point in points:
        if point[1] < a * point[0] + b:
            labels.append(1)
        else:
            labels.append(-1)
            
    labels = np.array(labels)     
    
    equation = (a,b)
    
    return list(zip(points, labels)), equation

#measure error on new data
def measure_accuracy(f, g):
    num = 1000
    points = []
    for i in range(num):
        coord = [random.uniform(-1,1), random.uniform(-1,1)]
        points.append(coord)
        
    points = np.array(points)
    
    #get true labels
    f_labels = []
    
    for point in points:
        if point[1] < f[0] * point[0] + f[1]:
            f_labels.append(1)
        else:
            f_labels.append(-1)

    g_labels = []
    
    for point in points:
        if point[1] < g[0] * point[0] + g[1]:
            g_labels.append(1)
        else:
            g_labels.append(-1)

    f_labels = np.array(f_labels)
    g_labels = np.array(g_labels)    
    
    #return percent of error
    return np.sum(f_labels != g_labels)/num

def log_reg(labelled_points):
    
    #logistic regression for 2d space
    weights = [0, 0, 0]

    nloops = 0
    
    while True:
        
        #store a copy of the old weights
        old_weights = weights
        random.shuffle(labelled_points)
        
        for point, label in labelled_points:

            point_data = np.hstack((np.array(1), point))

            #use current weights to generate hypothesis
            hypothesis = np.sign(np.matmul(weights, np.transpose(point_data)))
            gradient = -(label* point_data)/(1 + math.exp(label * hypothesis))

            #update weights
            weights = weights - eta * gradient

        #increment loop counter
        nloops += 1
        

        if np.linalg.norm(np.subtract(old_weights, weights)) < 0.01:
            #return number of loops used
            return weights, nloops

        
E_out = []
Epochs = []

for i in range(runs):
    labelled_points, equation = setup_problem(N)
    
    #train logistic regression
    weights, epochs = log_reg(labelled_points)
    
    E_out.append(measure_accuracy(equation, weights))
    Epochs.append(epochs)


print('Question 8')
print(np.average(E_out))

print('Question 9')
print(np.average(Epochs))



"""
Question 10

Again, by general set theory the union cannot be smaller than the largest set.
Showing that the union of the hypothesis set can be larger that the sum is a little bit tricky. However, by leveraging what we have already seen in the lectures,
we know that the VC dimension of the 2d perceptron is 3. However, it is clear that the VC dimension of the positive ray (i.e a one way perceptron) is only 1. Ergo, we have a case
where the VC dimension of the sum exceeds the sum of the VC dimension, and it cannot be that d is true.

"""

print('Question 10')

print("e")