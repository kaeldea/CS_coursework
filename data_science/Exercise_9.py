
# coding: utf-8

# 1. Import libraries and functions
# 2. Read and split the data

# In[1]:

import numpy as np
import math

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans


# In[2]:

#read the data
dataDigits = np.loadtxt('MNIST_179_digits.txt')

dataLabels = np.loadtxt('MNIST_179_labels.txt')


# Exercise 9
# 
# 1. Find 3 clusters of Digits Data

# In[3]:

#Cluster centers

#kmeans algorithm
kmeans = KMeans(n_clusters=3, n_init=10, max_iter=300, algorithm="full", random_state=19).fit(dataDigits) 
pred_classes = kmeans.predict(dataDigits)

#begin to find clusters
clusters = []
for clusterindex in range(3):
    #use data labels prior to splitting to find where the digits ended up
    cluster = dataLabels[np.where(pred_classes == clusterindex)]
    #print(cluster)
    clusters.append(cluster)

#Each cluster has a unique number of digits, count them separetly and give percentage 
for clusterindex in range(0,len(clusters)):
    unique, counts = np.unique(clusters[clusterindex], return_counts=True)
    percentages = []
    for i in range(0,counts.size):
        percentages.append((unique[i],float(counts[i])/float(sum(counts))*100));
    print(percentages)
    


# In[4]:

#single out lcusters
centers = kmeans.cluster_centers_

#reshape for visualization
center_0 = centers[0].reshape((28,28))
center_1 = centers[1].reshape((28,28))
center_2 = centers[2].reshape((28,28))

#plot cluster center 0
figure0 = plt.figure(figsize=(8,8))
ax0 = figure0.add_subplot(111)
ax0.imshow(center_0, cmap=plt.cm.binary)

plt.xticks([], [])
plt.yticks([], [])

plt.title('Cluster Center: Label 0')
plt.show();


# In[5]:

#cluster center 1
figure1 = plt.figure(figsize=(8,8))
ax1 = figure1.add_subplot(111)
ax1.imshow(center_1, cmap=plt.cm.binary)

plt.xticks([], [])
plt.yticks([], [])

plt.title('Cluster Center: Label 1')
plt.show();


# In[6]:

#plot cluster center 2
figure2 = plt.figure(figsize=(8,8))
ax2 = figure2.add_subplot(111)
ax2.imshow(center_2, cmap=plt.cm.binary)

plt.xticks([], [])
plt.yticks([], [])

plt.title('Cluster Center: Label 2')
plt.show();


# Part B
# 
# 1. Train a kNN classifier 

# In[7]:

#Split data into training and test sets: 80-20 split
train_900, test_200 = np.vsplit(dataDigits, [900])

#Split labels according to sets from above
labels_900, labels_200 = np.split(dataLabels, [900])


# In[8]:

#Prediction function taken from my previous assignment. T

#1. Find euclidean distance between training points and query points. 
#2. Use the distances to find the ‘k’ nearest neighbour(s) of the query points.
#3. Based on the class labels of these nearest neighbours, assign a class label to query points.

def predict(training_data, training_labels, test_data, k): 

    predicted_labels = [] #for all data 
    
    for idx_test, instance_test in enumerate(test_data): #must loop through index and features of test data
        
        closest_neighbors = [] #for instance_test
        closest_neighbor_distances = [] #no distances yet
        
        for idx_train, instance_train in enumerate(training_data): #loop through index and fetures of training data to find distances 
            
            #Get distances between test data and train data
            euclidean_distance = np.linalg.norm(np.array(instance_train) - np.array(instance_test))
            
            #comparison
            #Append distances up until the value of k
            if len(closest_neighbors) < k:
                closest_neighbors.append(idx_train) #where it is, helps to locate label
                closest_neighbor_distances.append(euclidean_distance) #add distances of closest neighbor 
            
            #Update list when shorter distances are found
            elif euclidean_distance < max(closest_neighbor_distances): #max is the highest value that was found thus far
                #Must find highest value to replace it with shorter values
                highest_distance_in_closest = max(closest_neighbor_distances)
                highest_distance_index = closest_neighbor_distances.index(highest_distance_in_closest) 
                
                closest_neighbor_distances[highest_distance_index] = euclidean_distance
                closest_neighbors[highest_distance_index] = idx_train
                
            
        #add closest neighbor by label to list
        #predicting labels of training data
        closest_neighbors_labels = []
        for neighbors in closest_neighbors:
            closest_neighbors_labels.append(training_labels[neighbors]) 
        
        #Mehtod of finding mode in python 2.7
        votes = closest_neighbors_labels
        mode_mj_vote = Counter(votes)
        predicted_labels.append(mode_mj_vote.most_common(1)[0][0]) #need to specify position otherwise it prints number of labels as well as most common label
        
    return predicted_labels


# In[9]:

#The following code was taken form my previous assignment.
#create indices for CV
cv = KFold(n_splits=5)

#loop over CV folds for test data
for num_neighbors in range(1,12, 2):
    
    scores = []
    
    average_loss = []
    
    for train, test in cv.split(train_900): #only use training data in the cross-validation process to generate folds 
        
        XTrainCV, XTestCV, YTrainCV, YTestCV = train_900[train], train_900[test], labels_900[train], labels_900[test]
    
        pred = predict(XTrainCV, YTrainCV, XTestCV, num_neighbors)
        
        #Accuracy score
        accuracy = accuracy_score(YTestCV, pred)
        scores.append(accuracy)
        
        #Average loss
        number_wrong = (1.0-accuracy)*len(XTestCV)
        average_loss.append(number_wrong)

        print("Accuracy score for " + str(num_neighbors) + " is " + str(accuracy) + " and number wrong is " + str(number_wrong))
    
    print("Average accuracy score for " + str(num_neighbors) + " is " + str(np.array(scores).mean()))
    
    print("Average loss for " + str(num_neighbors) + " is " + str(np.array(average_loss).mean()))


# In[10]:

#Split data into training and test sets: 80-20 split
train_900, test_200 = np.vsplit(dataDigits, [900])

#Split labels according to sets from above
labels_900, labels_200 = np.split(dataLabels, [900])

#Test data
#Accuracy score over test data
accTest = accuracy_score(labels_200, predict(train_900, labels_900, test_200, 1))
print("Accuracy (testing)" + str(accTest))

#Test self-coded algorithm against library funciton for test data
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(train_900, labels_900)
pred = neigh.predict(test_200)
print ("Accuracy score of Sklearn model for comparison (testing) " + str(accuracy_score(labels_200, pred)))

#Find actual number incorrectly predicted instances (average loss)
accuracy = accuracy_score(labels_200, pred)
number_wrong = (1.0-accuracy)*len(test_200)
print("Average loss " + str(number_wrong))


# In[ ]:



