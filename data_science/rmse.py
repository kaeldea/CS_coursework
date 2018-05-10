
# coding: utf-8

# In[9]:

import numpy as np
import math
from numpy.linalg import inv


# Exercise 3
# 
# RMSE is the standard deviation of the residuals (prediction errors).

# In[10]:

#red wine test dataset
data_train = np.loadtxt('redwine_training.txt')
data_test = np.loadtxt('redwine_testing.txt')


# Part A: run regression function on fixed acidity and report RMSE

# In[11]:

#Reusing code from exercise 2

#Separate fixed acidity and assign it as the input variable (independent)
#Separate wine quality and assign it as the output variable (dependent)
y = data_test[:,11]
one_feature = data_test[:,0]

#create vector of ones and concatenate fixed acidity
c1 = np.ones(shape=y.shape)[...]
x2 = np.array([c1,one_feature]).T

#run linear regression
final_c = inv(x2.transpose().dot(x2)).dot(x2.transpose()).dot(y)
print(final_c)


# Part B: run regression function on all features and report estimated weights

# In[12]:

#create vector of ones...
y = data_test[:,11]
features = np.delete(data_test, 11,axis=1)

v_2 = np.ones(shape=y.shape)[...,None]

x = np.concatenate((v_2, features), 1)

final_c2 = inv(x.transpose().dot(x)).dot(x.transpose()).dot(y)

print(final_c2)


# In[13]:

#create rmse function 
def rmse(predicted, y):
    differences = []
    #equation from exercise 3
    for i in range(0,len(y)):
        difference = np.abs(y[i] - predicted[i])**2
        differences.append(difference)
    #find mean
    sum_of_diffs = sum(differences)
    mean = sum_of_diffs / len(y)
    return np.sqrt(mean)


# In[14]:

#RMSE for one feature
y = data_test[:,11]

#combine ones with matrix
one_feature = np.array([features[:,0]]).T
ones = np.ones(shape=y.shape)[...,None]
one_feature_with_ones = np.hstack([ones,one_feature])

#make predictions
predictions = []
#start with one feature 
for wine in one_feature_with_ones:
    v = wine * final_c
    pred = sum(v)
    predictions.append(pred)
predictions1 = np.array(predictions)

#run predictions through rmse
rmse(predictions1, y)


# In[15]:

#RMSE for all features
y = data_test[:,11]
features = np.delete(data_test, 11,axis=1)

#make predictions
predictions = []
for wine in features:
    wine = np.concatenate([[1],wine])
    v = wine * final_c2 
    pred = sum(v)
    predictions.append(pred)
predictions = np.array(predictions)

#run predictions through rmse
rmse(predictions, y)


# In[ ]:



