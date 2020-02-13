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

for ans, N in [('a', 10), ('b', 25), ('c', 100), ('d', 500), ('e', 1000)]:
    if sigma**2*(1 - (d+1)/N) > 0.008:
        print(ans)
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
print('c')

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
        point_data = np.hstack((np.array(1), point))
        g_labels.append(np.matmul(g, np.transpose(point_data)))


    f_labels = np.array(f_labels)
    g_labels = np.array(g_labels)    
    

    #TODO: test error measure 
    #correct error measure is not classification error, must implement cross entropy error
    #return percent of error
    return 1/num * sum(math.log1p(1 + math.exp(-f_labels * g_labels)))

def log_reg(labelled_points):
    
    #logistic regression for 2d space
    weights = [0, 0, 0]

    nloops = 0
    
    while True:
        
        #store a copy of the old weights
        old_weights = weights

        #shuffle points randomly
        random.shuffle(labelled_points)
        
        for point, label in labelled_points:

            #add intercept term
            point_data = np.hstack((np.array(1), point))

            #use current weights to generate predicted value and gradient
            hypothesis = np.matmul(weights, np.transpose(point_data))
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

The perceptron learning algorithm is unique in that it only cares about fixing any points that are mislabelled, but any points
correctly labelled are effectively disregarded.
The only error measure where this is properly accounted for is e.


"""

print('Question 10')

print("e")