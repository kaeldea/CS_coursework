
# coding: utf-8

# In[1]:

import numpy as np

get_ipython().magic(u'pylab inline')
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

from collections import Counter


# In[2]:

#read the data
dataDigits = np.loadtxt('MNIST_179_digits.txt')

dataLabels = np.loadtxt('MNIST_179_labels.txt')

#Split data into training and test sets: 80-20 split
train_900, test_200 = np.vsplit(dataDigits, [900])

#Split labels according to sets from above
labels_900, labels_200 = np.split(dataLabels, [900])


# Use self implmented PCA from assignment 3

# In[3]:

#PCA function taken from my previous assignment
def pca(data):

    #Compute covariance matrix (alternative is scatter matrix) for the dataset
    cov_mat = np.cov(data, rowvar=False) #(np.cov(data.T))
    #print('Covariance Matrix: ' + str(cov_mat))

    #Eigenvectors and eigenvalues for the from the covariance matrix
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
    
    #Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
    
    
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    eig_vals_for_return = []
    eig_vecs_for_return = []
    
    for eig_pair in eig_pairs:
        eig_vals_for_return.append(eig_pair[0])
        eig_vecs_for_return.append(eig_pair[1])
    
    return(eig_vals_for_return, eig_vecs_for_return)


# Part A: cumulative variance

# In[4]:

#run PCA on training data
evals, evecs = pca(train_900)

#Create plot
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)

plt.title('Cumulative Variance over Number of Used Principal Components')
plt.xlabel('Number of used Principal Components')
plt.ylabel('Cumulative Normalized Variance (Eigenvalues)')

#cumulative, shows us percetnage of variance captured
#np.cumsum documentation: https://docs.scipy.org/doc/numpy/reference/generated/numpy.cumsum.html
c_var = np.cumsum(evals/np.sum(evals))

plt.plot(c_var)
plt.grid()
plt.show();

#cumulative, shows us percetnage of variance captured
print("90% variance: " + str(c_var[0:59]))


# Part B: Run k-Means algoritm, create 3 cluster centers as images and include percentage of each digit per cluster

# In[5]:

#Experiment with 20 components

#Perform PCA on centered data
evals, evecs = pca(train_900)
evecs = np.array(evecs)

#eigen_array for the first 20 principal components
eigen_array = np.zeros((20,784))
for i in range(20):
    eigen_array[i] = evecs[i]

#Data projected onto first principal component
matrix_w  = dataDigits.dot(eigen_array.T)

#Perform KMeans and add centroids to plot 
#startingpoint = np.vstack((matrix_w[0], matrix_w[1], matrix_w[2]))

kmeans = KMeans(n_clusters=3, n_init=10, max_iter=300, algorithm="full", random_state=19).fit(matrix_w) 
centers = (kmeans.cluster_centers_)

#Need to find the percentage of digits in each cluster
pred_classes = kmeans.predict(matrix_w)

#clusters 
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


# In[6]:

#Continue the process by plotting the images 

#matrix multiplication to transform the low-dimensional inputs back into high-dimensional space
cent = centers.dot(eigen_array)

#reshape to visualize digits
center_0 = cent[0].reshape((28,28))
center_1 = cent[1].reshape((28,28))
center_2 = cent[2].reshape((28,28))
 
#first center
figure0 = plt.figure(figsize=(8,8))
ax0 = figure0.add_subplot(111)
ax0.imshow(center_0, cmap=plt.cm.binary)
#get rid of tick marks
plt.xticks([], [])
plt.yticks([], [])
#plot title 
plt.title('Cluster Center: Label 0')
plt.show();

#second center
figure1 = plt.figure(figsize=(8,8))
ax1 = figure1.add_subplot(111)
ax1.imshow(center_1, cmap=plt.cm.binary)
#get rid of tick marks
plt.xticks([], [])
plt.yticks([], [])
#plot title 
plt.title('Cluster Center: Label 1')
plt.show();

#third center
figure2 = plt.figure(figsize=(8,8))
ax2 = figure2.add_subplot(111)
ax2.imshow(center_2, cmap=plt.cm.binary)
#get rid of tick marks
plt.xticks([], [])
plt.yticks([], [])
#create title 
plt.title('Cluster Center: Label 2')
plt.show();

#end of experiment with 20 components 


# In[7]:

#Experiment with 200 components

#Perform PCA on data
evals, evecs = pca(train_900)
evecs = np.array(evecs)
print(evecs.shape)

#eigen_array for the first 20 principal components
eigen_array = np.zeros((200,784))
for i in range(200):
    eigen_array[i] = evecs[i]

print(eigen_array.shape)

#Data projected onto first principal component
matrix_w  = dataDigits.dot(eigen_array.T)

#Perform KMeans and add centroids to plot 
startingpoint = np.vstack((matrix_w[0], matrix_w[1], matrix_w[2]))

kmeans = KMeans(n_clusters=3, init=startingpoint, n_init=1, max_iter=300, algorithm="full").fit(matrix_w) 
centers = (kmeans.cluster_centers_)

#Need to find the percentage of digits in each cluster
pred_classes = kmeans.predict(matrix_w)

#clusters
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




# In[8]:

#Continue the process by plotting the images 

#matrix multiplication to transform the low-dimensional inputs back into high-dimensional space
cent = centers.dot(eigen_array)

#reshape to visualize digits
center_0 = cent[0].reshape((28,28))
center_1 = cent[1].reshape((28,28))
center_2 = cent[2].reshape((28,28))
 
#first center
figure0 = plt.figure(figsize=(8,8))
ax0 = figure0.add_subplot(111)
ax0.imshow(center_0, cmap=plt.cm.binary)
#get rid of tick marks
plt.xticks([], [])
plt.yticks([], [])
#plot title 
plt.title('Cluster Center: Label 0')
plt.show();

#second center
figure1 = plt.figure(figsize=(8,8))
ax1 = figure1.add_subplot(111)
ax1.imshow(center_1, cmap=plt.cm.binary)
#get rid of tick marks
plt.xticks([], [])
plt.yticks([], [])
#plot title 
plt.title('Cluster Center: Label 1')
plt.show();

#third center
figure2 = plt.figure(figsize=(8,8))
ax2 = figure2.add_subplot(111)
ax2.imshow(center_2, cmap=plt.cm.binary)
#get rid of tick marks
plt.xticks([], [])
plt.yticks([], [])
#create title 
plt.title('Cluster Center: Label 2')
plt.show();

#end of experiment with 200 components 


# Part C: Classification, train a k-NN classifier 

# In[9]:

#Split data into training and test sets: 80-20 split
train_900, test_200 = np.vsplit(dataDigits, [900])

#Split labels according to sets from above
labels_900, labels_200 = np.split(dataLabels, [900])


# In[10]:

#Self-implemented algorithm from previous assignment

#KNN classifier 
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


# In[11]:

#Perform PCA on centered data
evals, evecs = pca(train_900)
evecs = np.array(evecs)

#eigen_array for the first 20 principal components
eigen_array = np.zeros((20,784))
for i in range(20):
    eigen_array[i] = evecs[i]

#Data projected onto first principal component
matrix_w  = dataDigits.dot(eigen_array.T)

#create indices for CV
cv = KFold(n_splits=5)

#taken from my previous assignment
#loop over CV folds for test data
for num_neighbors in range(1,12, 2):
    
    scores = []
    
    average_loss = []
    
    for train, test in cv.split(matrix_w): #only use training data in the cross-validation process to generate folds 
        
        XTrainCV, XTestCV, YTrainCV, YTestCV = matrix_w[train], matrix_w[test], dataLabels[train], dataLabels[test]
    
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


# In[12]:

#First 20 PC

#Split data into training and test sets: 80-20 split
train_m, test_m = np.vsplit(matrix_w, [900])

#Split labels according to sets from above
labels_m, labels_m_2 = np.split(dataLabels, [900])

#Accuracy score over test data
accTest = accuracy_score(labels_m_2, predict(train_m, labels_m, test_m, 1))
print("Accuracy (testing)" + str(accTest))

#Test self-coded algorithm against library funciton for test data
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(train_m, labels_m)
pred = neigh.predict(test_m)
print ("Accuracy score of Sklearn model for comparison (testing) " + str(accuracy_score(labels_m_2, pred)))


# In[13]:

#First 200 PC

#Split data into training and test sets: 80-20 split
train_900, test_200 = np.vsplit(dataDigits, [900])

#Split labels according to sets from above
labels_900, labels_200 = np.split(dataLabels, [900])

#Perform PCA on centered data
evals, evecs = pca(train_900)
evecs = np.array(evecs)

#eigen_array for the first 20 principal components
eigen_array = np.zeros((200,784))
for i in range(20):
    eigen_array[i] = evecs[i]

#Data projected onto first principal component
matrix_w  = dataDigits.dot(eigen_array.T)

#create indices for CV
cv = KFold(n_splits=5)

#taken from my previous assignment 
#loop over CV folds for test data
for num_neighbors in range(1,12, 2):
    
    scores = []
    
    average_loss = []
    
    for train, test in cv.split(matrix_w): #only use training data in the cross-validation process to generate folds 
        
        XTrainCV, XTestCV, YTrainCV, YTestCV = matrix_w[train], matrix_w[test], dataLabels[train], dataLabels[test]
    
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


# In[14]:

#First 200 PC

#Split data into training and test sets: 80-20 split
train_m, test_m = np.vsplit(matrix_w, [900])

#Split labels according to sets from above
labels_m, labels_m_2 = np.split(dataLabels, [900])

#Accuracy score over test data
accTest = accuracy_score(labels_m_2, predict(train_m, labels_m, test_m, 1))
print("Accuracy (testing)" + str(accTest))

#Test self-coded algorithm against library funciton for test data
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(train_m, labels_m)
pred = neigh.predict(test_m)
print ("Accuracy score of Sklearn model for comparison (testing) " + str(accuracy_score(labels_m_2, pred)))


# In[ ]:



