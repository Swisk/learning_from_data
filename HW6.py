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
import pandas as pd

'''
Question 1
Since H' is a subset of H, the approximation capability of H' should be worse. This implies that there will be more deterministic noise.
'''
print('Question 1')
print('b')

"""
Question 2

"""

print('Question 2')
#set up question
df_train_raw = pd.read_csv(r'c:/Users/shane/Documents/Tech Dev/learning_from_data/in.dta.txt', sep = '\s+', names=('x1', 'x2', 'y'))
X = df_train_raw.loc[: , 'x1': 'x2']
y = df_train_raw['y']

X['1'] = 1
X['x1^2'] = X['x1']**2
X['x2^2'] = X['x2']**2
X['x1x2'] = X['x1']*X['x2']
X['|x1-x2|'] = abs(X['x1'] - X['x2'])
X['|x1+x2|'] = abs(X['x1'] + X['x2'])


#building linear regression
#one step learning
xprimex = X.transpose().dot(X)
inv = pd.DataFrame(np.linalg.pinv(xprimex.values), xprimex.columns, xprimex.index)

weights = inv.dot(X.transpose()).dot(y)

E_in = np.sum(y != np.sign(weights.dot(X.transpose())))/len(y)

df_test = pd.read_csv(r'c:/Users/shane/Documents/Tech Dev/learning_from_data/out.dta.txt', sep = '\s+', names=('x1', 'x2', 'y'))
X_test = df_test.loc[: , 'x1': 'x2']
X_test['1'] = 1
X_test['x1^2'] = X_test['x1']**2
X_test['x2^2'] = X_test['x2']**2
X_test['x1x2'] = X_test['x1']*X_test['x2']
X_test['|x1-x2|'] = abs(X_test['x1'] - X_test['x2'])
X_test['|x1+x2|'] = abs(X_test['x1'] + X_test['x2'])

y_test = df_test['y']

df_test['1'] = 1


E_out = np.sum(y_test != np.sign(weights.dot(X_test.transpose())))/len(y_test)

print(E_in, E_out)
print('a')

"""
Question 3

Modify solution for weights to take into account the weight decay
"""

print('Question 3')
k = -3

inv_decayed = pd.DataFrame(np.linalg.pinv(xprimex.values + 10**k * np.identity(8)), xprimex.columns, xprimex.index)
weights_decayed = inv_decayed.dot(X.transpose()).dot(y)

E_in = np.sum(y != np.sign(weights_decayed.dot(X.transpose())))/len(y)
E_out = np.sum(y_test != np.sign(weights_decayed.dot(X_test.transpose())))/len(y_test)

print(E_in, E_out)
print('d')

"""
Question 4
Rerun the same calculations with a different k

"""

print('Question 4')
k = 3

inv_decayed = pd.DataFrame(np.linalg.pinv(xprimex.values + 10**k * np.identity(8)), xprimex.columns, xprimex.index)
weights_decayed = inv_decayed.dot(X.transpose()).dot(y)

E_in = np.sum(y != np.sign(weights_decayed.dot(X.transpose())))/len(y)
E_out = np.sum(y_test != np.sign(weights_decayed.dot(X_test.transpose())))/len(y_test)
print(E_in, E_out)
print('e')

"""
Question 5

Loop over all possiblities
"""

print('Question 5')

for k in [-2, -1, 0, 1 ,2]:

    inv_decayed = pd.DataFrame(np.linalg.pinv(xprimex.values + 10**k * np.identity(8)), xprimex.columns, xprimex.index)
    weights_decayed = inv_decayed.dot(X.transpose()).dot(y)

    E_in = np.sum(y != np.sign(weights_decayed.dot(X.transpose())))/len(y)
    E_out = np.sum(y_test != np.sign(weights_decayed.dot(X_test.transpose())))/len(y_test)

    print(k, E_in, E_out)


print('d')

"""
Question 6
We can find this answer by observing the lowest out of sample error amongst integer k. We expect
this to be bounded such that too large K causes an increase in out of sample error, so the 
mininum should be a middle point.

"""

print('Question 6')
lowest = (10, 0)

for k in range(-10, 10):

    inv_decayed = pd.DataFrame(np.linalg.pinv(xprimex.values + 10**k * np.identity(8)), xprimex.columns, xprimex.index)
    weights_decayed = inv_decayed.dot(X.transpose()).dot(y)

    E_out = np.sum(y_test != np.sign(weights_decayed.dot(X_test.transpose())))/len(y_test)

    if E_out < lowest[0]:
        lowest = (E_out, k)

print(lowest[0])
print('b')

"""
Question 7

"""

print('Question 7')
print('c')

"""
Question 8

"""

print('Question 8')
print('d')

"""
Question 9

"""

print('Question 9')
print('a')



"""
Question 10


"""

print('Question 10')

print("e")