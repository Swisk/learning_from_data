# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 17:05:12 2018

@author: Shane
"""

import math

error = 0.05
N = 480000
print(4* (2*N)**10 * math.exp(-(1/8)*error**2 * N))


"""
Question 2
"""

d=50
sigma=.05
N=10000

print(math.sqrt((8/N)*math.log((4*(2*N)**d) /sigma)))
print(math.sqrt((2/N)*math.log((2*N*(N)**d))) + math.sqrt((2/N)*math.log(1/sigma)) + 1/N)
print(math.sqrt((1/N) * math.log(6/sigma * (2*N)**d)))
print(math.sqrt(1/(2*N) * (math.log(4/sigma) + math.log((N**2)**d))))

