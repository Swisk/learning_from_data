# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 16:02:27 2018

@author: Shane
"""

import random
import math
import numpy as np

'''
Question 1

run the experiment

'''
# flips N number of coins 10 times
def coin_flip(N):
    
    result = [None] * N
    for coin in range(N):
        heads = 0
        
        #result represents number of heads
        for i in range(10):
            heads += random.randint(0,1)
            
        #add number of heads to array
        result[coin] = heads
            
    return result
        
def flip_experiment(runs):
    v1 = [None] * runs
    vrand = [None] * runs
    vmin = [None] * runs
    for run in range(runs):
        #set up experiment parameters
        coins = 1000
        random_index = random.randrange(0, coins)
        
        run_result = coin_flip(coins)
        
        v1[run] = (run_result[0])
        vrand[run] = (run_result[random_index])
        vmin[run] = (min(run_result))
        
    return v1, vrand, vmin

#runs cannot run finish
runs = 10000

v1, vrand, vmin = flip_experiment(runs)

print(np.average(vmin))

'''
It is very likely that only the random coins meet the inequality, as the minimum coin is obviously biased.

'''

#average is 0.5
#convert to fractions 
vmin = vmin /10
#calculate distribution of values within error
min_error = abs(vmin - 0.5)

#calculate hoeffding's threshold, for some arbitrary error
error = 0.01

exceed_prob = len(np.where(min_error > error)) / len(min_error)
thresh = 2 * math.exp(-2*error**2 * 100000)
