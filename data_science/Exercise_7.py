
# coding: utf-8

# In[1]:

import numpy as np
import math
import matplotlib.pyplot as plt
import pylab
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression


# In[2]:

#read the data
iris1_test = np.loadtxt('Iris2D1_test.txt')
iris1_train = np.loadtxt('Iris2D1_train.txt')

iris2_test = np.loadtxt('Iris2D2_test.txt')
iris2_train = np.loadtxt('Iris2D2_train.txt')


# Part A

# In[3]:

#Iris train data 2D1
#Separate into label and x,y coordinates
iris1_tra_m = np.delete(iris1_train, 2,axis=1)
iris1_tra_l = iris1_train[:,2]
#create plot
colors = iris1_tra_l
figure = plt.figure(figsize=(8,8))
ax = figure.add_subplot(111)
ax.scatter(iris1_tra_m[:,0], iris1_tra_m[:,1], c=colors)

#create title
plt.title('Iris2D1 Train')

#create legend
ax.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
purple_patch = mpatches.Patch(color='purple', label='Label 0')
plt.legend(handles=[purple_patch])

yellow_patch = mpatches.Patch(color='yellow', label='Label 1')
plt.legend(handles=[purple_patch, yellow_patch])

plt.show();

#Iris test data 2D1
#Separate into label and x,y coordinates
iris1_te_m = np.delete(iris1_test, 2,axis=1)
iris1_te_l = iris1_test[:,2]

#create plot
figure = plt.figure(figsize=(8,8))
ax = figure.add_subplot(111)

colors = iris1_te_l
ax.scatter(iris1_te_m[:,0], iris1_te_m[:,1], c=colors)

#create title
plt.title('Iris2D1 Test')

#create legend
ax.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
purple_patch = mpatches.Patch(color='purple', label='Label 0')
plt.legend(handles=[purple_patch])

yellow_patch = mpatches.Patch(color='yellow', label='Label 1')
plt.legend(handles=[purple_patch, yellow_patch])

plt.show();


# In[4]:

#Iris train data 2D2
#Separate into label and x,y coordinates
iris2_tra_m = np.delete(iris2_train, 2,axis=1)
iris2_tra_l = iris2_train[:,2]
#create plot
figure = plt.figure(figsize=(8,8))
ax = figure.add_subplot(111)
colors = iris2_tra_l
ax.scatter(iris2_tra_m[:,0], iris2_tra_m[:,1], c=colors)

#create title
plt.title('Iris2D2 Train')

#create legend
ax.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
purple_patch = mpatches.Patch(color='purple', label='Label 0')
plt.legend(handles=[purple_patch])

yellow_patch = mpatches.Patch(color='yellow', label='Label 1')
plt.legend(handles=[purple_patch, yellow_patch])

plt.show();

#Iris test data 2D2
#Separate into label and x,y coordinates
iris2_te_m = np.delete(iris2_test, 2,axis=1)
iris2_te_l = iris2_test[:,2]

#create plot
figure = plt.figure(figsize=(8,8))
ax = figure.add_subplot(111)

colors = iris2_te_l
ax.scatter(iris2_te_m[:,0], iris2_te_m[:,1], c=colors)

#create title
plt.title('Iris2D2 Test')

#create legend
ax.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
purple_patch = mpatches.Patch(color='purple', label='Label 0')
plt.legend(handles=[purple_patch])

yellow_patch = mpatches.Patch(color='yellow', label='Label 1')
plt.legend(handles=[purple_patch, yellow_patch])

plt.show();


# Part B

# In[5]:

#create logistic regression funciton like in lecture
#additional ispiration from https://beckernick.github.io/logistic-regression-from-scratch/

#need sigmoid function to transform the linear model of the predictors
def sigmoid(scores):
    return 1/(1+np.exp(-scores))

#log-likelihood (e.g. sum over input data) 
def log_likelihood(features,labels,weights):
    scores = np.dot(features,weights)
    return np.sum( labels*scores - np.log(1+np.exp(scores)))

#take its derivative for gradient descent implementation 
def likelihood_derivative(features, loss):
    return np.dot(features.T, loss)

def logistic_reg(training_set, training_labels, num_iter, learning_rate):
    
    #empty, fill in as it iterates
    intercept = np.ones((training_set.shape[0], 1))
    training_set = np.hstack((intercept, training_set))
    
    #empty, fill in as it iterates
    weights = np.zeros(training_set.shape[1])
    
    for step in range(num_iter):
        
        #start preditions
        predictions = sigmoid(np.dot(training_set,weights))
        
        #calculate loss, similar to last exercise
        loss = training_labels - predictions
        #find direction to "step"
        direction = (likelihood_derivative(training_set, loss))
        #updates weights (similar to last exercise)
        weights = weights + learning_rate*direction
    
    return weights


# Part C and Part D

# In[6]:

#Run iris data through logisitc regression 
model_iris2d1 = logistic_reg(iris1_tra_m, iris1_tra_l, 1000000, 2e-5)
model_iris2d2 = logistic_reg(iris2_tra_m, iris2_tra_l, 1000000, 2e-5)
model_iris2d1_test = logistic_reg(iris1_te_m, iris1_te_l, 1000000, 2e-5)
model_iris2d2_test = logistic_reg(iris2_te_m, iris2_te_l, 1000000, 2e-5)

print "iris2d1 train " + str(model_iris2d1)
print "iris2d2 train " + str(model_iris2d2)
print "iris2d1 test " + str(model_iris2d1_test)
print "iris2d2 test " + str(model_iris2d2_test)


# In[7]:

#Find test error for Iris 2D2 Test

def find_error(data, labels, model):
    #put through sklearn
    skr = LogisticRegression(fit_intercept=True)
    skr.fit(data, labels)

    data_int = np.hstack((np.ones((data.shape[0], 1)), data)) #add intercept to data, for pred
    scores = np.dot(data_int, model) #call final weights
    predictions = np.round(sigmoid(scores)) #sigmoid for final prediction

    #find test error 
    accuracy = ((sum(predictions == labels).astype(float) / len(predictions)))
    error = 1-(accuracy)

    accuracy_sklearn = (skr.score(data, labels))
    error_sklearn = 1-(accuracy_sklearn)

    print 'Error from self-implemented: ' + str(error)
    print 'Error from sklearn: ' + str(error_sklearn)


print 'train2D1'
find_error(iris1_tra_m, iris1_tra_l, model_iris2d1)
print 'train2D2: '
find_error(iris2_tra_m, iris2_tra_l, model_iris2d2)
print 'test2D1: '
find_error(iris1_te_m, iris1_te_l, model_iris2d1_test)
print 'test2D2: '
find_error(iris2_te_m, iris2_te_l, model_iris2d2_test)


# In[ ]:



