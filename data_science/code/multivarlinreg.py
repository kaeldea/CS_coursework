
# coding: utf-8

# In[1]:

import numpy as np
import math
from numpy.linalg import inv


# In[5]:

#red wine train dataset
data_train = np.loadtxt('redwine_training.txt')
data_test = np.loadtxt('redwine_testing.txt')


# Multivariate linear regression can be computed consciely
# 
# The linear model t=f(x,w) can be expressed as Y=XC with X being the input variables, C being (X.T * X)^-1 * X.T * y where y is the output variables 

# Part A: run regression function on fixed acidity and report the weights

# In[6]:

#Separate fixed acidity and assign it as the input variable (independent)
#Separate wine quality and assign it as the output variable (dependent)
y = data_train[:,11]
one_feature = data_train[:,0]

#create vector of ones and concatenate fixed acidity
v_1 = np.ones(shape=y.shape)[...]
x2 = np.array([v_1,one_feature]).T

#run linear regression
final_c = inv(x2.transpose().dot(x2)).dot(x2.transpose()).dot(y)
print(final_c)


# Part B: run regression function on all features and report estimated weights

# In[8]:

#https://datascienceplus.com/linear-regression-from-scratch-in-python/
#https://www.hackerearth.com/practice/machine-learning/linear-regression/multivariate-linear-regression-1/tutorial/

#Separate all features and assign them as the input variable (independent)
#Separate wine quality and assign it as the output variable (dependent)
y = data_train[:,11]
features = np.delete(data_train, 11,axis=1)

#create vector of ones and concatenate all features
v_2 = np.ones(shape=y.shape)[...,None]
x = np.concatenate((v_2, features), 1)

#run linear regression
final_c2 = inv(x.transpose().dot(x)).dot(x.transpose()).dot(y)
print(final_c2)


# In[ ]:



