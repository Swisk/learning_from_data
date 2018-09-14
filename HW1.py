# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 17:50:24 2018

@author: Shane
"""
import random
import numpy as np


N = 10
d = 2
runs = 10000

#function that implements the PLA learning algorithm
def PLA(data, labels):
    
    nloops = 0
    weights = [[0] * (d + 1)]
    
    while True:
        
        #increment loop counter
        nloops += 1
        
        if nloops > 10000:
            #return number of loops used
            return nloops
        
        #use current weights to generate hypothesis
        hypothesis = np.sign(np.matmul(weights, np.transpose(data)))
        
        wrong = labels != hypothesis
        test = np.where(wrong)
        
        #if there are misclassified points
        if len(test[1]) >= 1:
            #select one at random
            select = random.choice(test[1])
            #update weights
            weights = weights + labels[select] * data[select]

        else:
            #return number of loops used
            return nloops, weights

#setup for PLA hw problem
# generate points
def setup_problem(N):
    points = []
    for i in range(N):
        coord = [random.uniform(-1,1), random.uniform(-1,1)]
        points.append(coord)
        
    points = np.array(points)
    
    intercept = np.ones((N,1))
    
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
    
    points = np.hstack((intercept, points))
    
    equation = (a,b)
    
    return points, labels, equation

#measure probabilty of accuracy
def measure_accuracy(line_equation):
    pass
    
    

#try running PLA
run_loops = []

for i in range(runs):
    points, labels, equation = setup_problem(N)
    nloops, weights = PLA(points, labels)
    run_loops.append(nloops)

#average number of runs
print(np.mean(run_loops))


#measure probabilty of 