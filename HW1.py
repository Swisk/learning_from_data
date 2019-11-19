# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 17:50:24 2018

@author: Shane
"""
import random
import numpy as np

'''
Question 1

d
'''

'''
Question 2

ii and iv are true
hence  a
'''
'''
Question 3

This question is asking about proper use of priors. Given that a black ball was selected, it is twice as likely that bag with 2 black
balls in it was selected, as the other bag has a chance of revealing a white ball.

Hence, there is 2/3 chance the other ball is black
d
'''
'''
Question 4

This probability is simply the chance of not red, ten times in a row
Evaluating, the answer is therefore b.
'''
print(0.45**10)



'''
Question 5

Wee identified the probability of v=0 in the previous question. Over 1000 samples, the probability that at least one
sample has case v=0 is the complement of all samples not having v=0.

The answer is c.
'''
v=0.45**10

#all having v is (1-v)**1000
print(1-(1-v)**1000)


'''
Question 6

Observe the question is merely asking how many points did it get right on the unseen data as the score in a mathematical way.

There is no need to evaluate this explicitly - purely by symmetry, as all combinations of points are covered, any hypothesis must
be able to match some and not match others equally.

This is because the generation of unseen points does not need to follow any trend ovserved within D, and occurs purely 
combinatorily.

The answer must be e.

'''



'''
Question 7-10

We run the function as given, changing N as necessary.

answers are:
b
c
b
b
'''

N = 100
d = 2
runs = 1000

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
def measure_accuracy(weights, g):
    num = 9999
    points = []
    for i in range(num):
        coord = [random.uniform(-1,1), random.uniform(-1,1)]
        points.append(coord)
        
    points = np.array(points)
    
    #get true labels
    g_labels = []
    
    for point in points:
        if point[1] < g[0] * point[0] + g[1]:
            g_labels.append(1)
        else:
            g_labels.append(-1)
    
    #get predicted labels
    intercept = np.ones((num, 1))
    points = np.hstack((intercept, points))
    
    f_labels = np.sign(np.matmul(weights, np.transpose(points)))
    
    
    f_labels = np.array(f_labels)
    g_labels = np.array(g_labels)    
    
    #return percent accurate
    return np.sum(f_labels == g_labels)/num
    

#try running PLA
run_loops = []
acc_loops = []

for i in range(runs):
    points, labels, equation = setup_problem(N)
    nloops, weights = PLA(points, labels)
    acc = measure_accuracy(weights, equation)
    run_loops.append(nloops)
    acc_loops.append(acc)

#average number of runs
print(np.mean(run_loops))
print(1-np.mean(acc))
