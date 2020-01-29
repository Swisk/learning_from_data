# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 17:05:12 2018

@author: Shane
"""

import math
import scipy.integrate as integrate
import random
import numpy as np

'''
Question 1
'''

#plug and play into formula 


print('Question 1')
error = 0.05
M = 1

for i, N in enumerate([500,1000,1500,2000]):
    ans = 2* M * math.exp(-2*error**2 * N)
    if ans < 0.03:
        print(['a', 'b', 'c', 'd'][i])
        break


"""
Question 2
"""

print('Question 2')
M = 10

for i, N in enumerate([500,1000,1500,2000]):
    ans = 2* M * math.exp(-2*error**2 * N)
    if ans < 0.03:
        print(['a', 'b', 'c', 'd'][i])
        break


"""
Question 3

same as question 2 but with different M
"""

print('Question 3')

M = 100

for i, N in enumerate([500,1000,1500,2000]):
    ans = 2* M * math.exp(-2*error**2 * N)
    if ans < 0.03:
        print(['a', 'b', 'c', 'd'][i])
        break

"""
Question 4

The perceptron model in 3 dimensional space must have a larger break point than that in 2 dimensional space, as there is another dimension to use to separate the points.
However, this fails when there are 5 points, as regardless of the arrangement of points it must be the case that there are "opposing" points rather than "adjacent". 

"""

print('Question 4')
print('b')

"""
Question 5

By evaluating based on the constraint that the growth function must be less than 2**N, we see that all 5 options are possible.
A further restiction is that the resulting functions must be polynomial in N, such that the summation of N choose i until k-1 holds.
This excludes iii and iv.
"""

print('Question 5')
print('b')


"""
Question 6

We can see that a case where the dichotomy of the middle point being -1 cannot be created.

"""

print('Question 6')
print('c')


"""
Question 7

The intepretation of this also makes sense, as such a dichotomy could be expressed as choosing 4 out of N + 1 spaces between points.
The only option that fits this case is b.

"""

print('Question 7')

print("C")


"""
Question 8

From 6, we know that a break points exists at 5 for M = 2. This rules out options A, B, C, E.
D also makes sense, as such a hypothesis set only fails when there are M + 1 non adjacent points to be classified at +1. 
This requires at least M points as separators.

"""

print('Question 8')

print("d")

"""
Question 9

A similar thought process is required to that of the M interval hypothesis set. The triangle may swell to capture any adjacent points,
so the minimum amount of points that cannot be shattered would require 4 distinct non adjacent points to be +1. This requires 8 points in total.

"""

print('Question 9')

print("d")

"""
Question 10
The concentric circle is just a fancy implementation of the interval hypothesis set in 3d.

"""

print('Question 10')

print("b")