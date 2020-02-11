# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 17:50:24 2018

code split into multiple parts for ease of running

@author: Shane
"""
import random
import numpy as np
from sklearn import linear_model

'''
Question 5-7

Running it experimentally to get the output

N = 100
c
c

change N = 10
a
'''
N = 10
d = 2
runs = 1000

#function that implements the PLA learning algorithm
def PLA(data, labels, weights = [0] * (d + 1)):
    
    intercept = np.ones((N,1))
    data = np.hstack((intercept, data))
    
    nloops = 0
    
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
        if len(test[0]) >= 1:
            #select one at random
            select = random.choice(test[0])
            #update weights
            weights = weights + labels[select] * data[select]

        else:
            #return number of loops used
            return nloops, weights

#setup for PLA hw problem

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
    
    return points, labels, equation

#function to measure accuracy for out of sample (newly generated) data
def measure_accuracy(f, model):
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
    
    g_labels = np.sign(model.predict(points))
    
    f_labels = np.array(f_labels)
    g_labels = np.array(g_labels)    
    
    #return percent accurate
    return np.sum(f_labels == g_labels)/num
    

#try running PLA
acc_in = []
acc_out = []
nruns = []

for i in range(runs):
    points, labels, equation = setup_problem(N)
    
    #train linear regression
    regr = linear_model.LinearRegression()
    regr.fit(points, labels)
    
    y_pred_in = np.sign(regr.predict(points))
    
    acc_in.append(np.sum(labels == y_pred_in)/len(labels))

    acc_out.append(measure_accuracy(equation, regr))
    
    #PLA loops
    nloops, weights = PLA(points, labels, np.insert(regr.coef_, 0, [regr.intercept_]))
    nruns.append(nloops)


#average number of runs
print(np.mean(acc_in))
print(np.mean(acc_out))
print(np.mean(nruns))
