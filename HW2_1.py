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

answer is closest to b.
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

#runs cannot run finish, so reduced in magnitude
runs = 1000

v1, vrand, vmin = flip_experiment(runs)

#average is 0.5
#convert to fractions 
v1 = [(lambda x: x/10)(x) for x in v1]
vrand = [(lambda x: x/10)(x) for x in vrand]
vmin = [(lambda x: x/10)(x) for x in vmin]

print(np.average(vmin))

'''
Question 2

It is very likely that only the random coins meet the inequality, as the minimum coin is obviously biased. 
Hence, the answer should be d.

'''

'''
Question 3

e. Since it is a binary function, we need to account for the case where it corrects the noise. 
'''

'''
Question 4

b. At this value mu will cancel out.
'''