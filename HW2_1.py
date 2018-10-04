# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 16:02:27 2018

@author: Shane
"""

import random
import math
import numpy as np

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

runs = 100000
v1, vrand, vmin = flip_experiment(runs)

#calculate hoeffding's 
error = 0.01
thresh = 2 * math.exp(-2*error**2 * 100000)
