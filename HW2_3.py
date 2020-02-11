# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 17:50:24 2018

@author: Shane
"""
import random
import numpy as np
from sklearn import linear_model

'''
Question 8-10

running it experimentally

d

e
'''
N = 1000
d = 2
runs = 1000


def setup_problem(N):
    points = []
    for i in range(N):
        coord = [random.uniform(-1,1), random.uniform(-1,1)]
        points.append(coord)
        
    points = np.array(points)
    
    
    #label points
    labels = []
    for point in points:
        true_label = np.sign(point[0]**2 + point[1]**2 - 0.6)

        #adding noise
        if np.random.uniform() < 0.9:
            labels.append(true_label)
        else:
            labels.append(np.negative(true_label))
            
    labels = np.array(labels)     
    
    return points, labels

#measure probabilty of accuracy
def measure_accuracy(model):
    num = 1000
    points = []
    for i in range(num):
        coord = [random.uniform(-1,1), random.uniform(-1,1)]
        points.append(coord)
        
    points = np.array(points)
    
    #get true labels
    f_labels = []
    
    for point in points:
        true_label = np.sign(point[0]**2 + point[1]**2 - 0.6)
        if np.random.uniform() < 0.9:
            f_labels.append(true_label)
        else:
            f_labels.append(np.negative(true_label))
            
    points2 = np.hstack((points, points ** 2, (points[:, 0] * points [:, 1])[:, None]))
    
    g_labels = np.sign(model.predict(points2))
    
    f_labels = np.array(f_labels)
    g_labels = np.array(g_labels)    
    
    #return percent accurate
    return np.sum(f_labels == g_labels)/num
    

#try running 
acc_pre = []
acc_post = []
acc_out = []
weights = []

for i in range(runs):
    points, labels = setup_problem(N)
    
    #train linear regression
    regr = linear_model.LinearRegression()
    regr.fit(points, labels)
    
    y_pred_pre = np.sign(regr.predict(points))
    
    acc_pre.append(np.sum(labels == y_pred_pre)/len(labels))
    
    #train linear regression (with transforms)
    regr2 = linear_model.LinearRegression()
    points2 = np.hstack((points, points ** 2, (points[:, 0] * points [:, 1])[:, None]))
    regr2.fit(points2, labels)
    
    weights.append(np.insert(regr2.coef_, 0, [regr2.intercept_]))
    
    y_pred_post = np.sign(regr2.predict(points2))
    acc_post.append(np.sum(labels == y_pred_post)/len(labels))
    
    acc_out.append(measure_accuracy(regr2))


#average number of runs
weights = np.array(weights)
print(np.mean(acc_pre))
print(weights.mean(0))
print(np.mean(acc_out))
