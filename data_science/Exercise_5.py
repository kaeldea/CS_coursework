
# coding: utf-8

# In[1]:

import numpy as np
import math

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Exercise 5

# In[2]:

#Load datasets

#pesticide dataset
dataTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
dataTest = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')

#split crop data into input variables and labels
XTrain = dataTrain[:, :-1] #13 dimensions without label
YTrain = dataTrain[:, -1] #only labels

XTest = dataTest[:, :-1] #13 dimensions without label
YTest = dataTest[:, -1] #only labels


# In[3]:

#Create a random forest Classifier with 50 trees
rfc = RandomForestClassifier()

#Train the Classifier to take the training features and learn how they relate to the training y (the species)
rfc.fit(XTrain, XTrain)


# In[4]:

#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
#http://dataaspirant.com/2017/06/26/random-forest-classifier-python-scikit-learn/
#http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

#Create a random forest Classifier with 50 trees
def random_forest_classifier(features, target):
    rfc = RandomForestClassifier(n_estimators=50)
    rfc.fit(features, target)
    return rfc

#Train the Classifier to take the training features and learn how they relate to the training_y 
trained_model = random_forest_classifier(XTrain, YTrain)
print("Classifier for Train Dataset: " + str(trained_model))

#Test the classifier on XTest and YTest
predictions = trained_model.predict(XTest)
#print("Classifier for Test Dataset: " + str(predictions))

print "Train Accuracy: ", accuracy_score(YTrain, trained_model.predict(XTrain))
print "Test Accuracy: ", accuracy_score(YTest, predictions)


# In[ ]:



