
# coding: utf-8

# 1. Import libraries and functions
# 2. Read and split the data

# In[1]:

import numpy as np

get_ipython().magic('pylab inline')
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


# In[2]:

#read murder data
data = np.loadtxt('murderdata2d.txt')

#pesticide dataset
dataTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
dataTest = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')

#split crop data into input variables and labels
XTrain = dataTrain[:, :-1] #13 dimensions without label
YTrain = dataTrain[:, -1] #only labels

XTest = dataTest[:, :-1] #13 dimensions without label
YTest = dataTest[:, -1] #only labels


# Exercise 1

# Part A
# 
# 1) Create a function that returns eigenvalues and eigenvectors in descending order

# In[3]:

#Inspiration: http://sebastianraschka.com/Articles/2014_pca_step_by_step.html
#Inspiration from PCA_lecture_handout given by lecturers

def pca(data):

    #Compute covariance matrix (alternative is scatter matrix) for the dataset
    cov_mat = np.cov(data, rowvar=False) #(np.cov(data.T))
    #print('Covariance Matrix: ' + str(cov_mat))

    #Eigenvectors and eigenvalues for the from the covariance matrix
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
    
    #Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
    #print("pairs: " + str(eig_pairs))
    
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    eig_vals_for_return = []
    eig_vecs_for_return = []
    
    for eig_pair in eig_pairs:
        eig_vals_for_return.append(eig_pair[0])
        eig_vecs_for_return.append(eig_pair[1])
    
    return(eig_vals_for_return, eig_vecs_for_return)



# Part B
# 
# 1) Create a scatterplot 
# 
# 2) Add to the scatterplot the mean and the principal eigenvectors pointing out of them

# In[4]:

evals, evecs = pca(data)

#Scale eigenvectors by SD of their eigenvalues 
mean_vector1 = [float(sum(n))/len(n) for n in zip(*data)]

s0 = np.sqrt(evals[0])
s1 = np.sqrt(evals[1])

#centering data
one = np.array(data[:,0] - mean_vector1[0])
two = np.array(data[:,1] - mean_vector1[1])

data1 = np.column_stack((one, two))

mean_vector = [float(sum(n))/len(n) for n in zip(*data1)]

#Create plot: manipulate size, add text, add title, labels and legend 
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.axis('equal')

plt.title('Variance Found Over Murder Dataset')
plt.xlabel('Percent Unemployed')
plt.ylabel('Murders per Annum (per 1,000,000 Inhabitants)')

#Scatter plot: data, mean, eigenvectors 
plt.scatter(one, two)
ax.plot([mean_vector[0], s0*evecs[0][0]+mean_vector[0]], [mean_vector[1], s0*evecs[0][1]+mean_vector[1]], 'r')
ax.plot([mean_vector[0], s1*evecs[1][0]+mean_vector[0]], [mean_vector[1], s1*evecs[1][1]+mean_vector[1]],'r')                                                                         
ax.plot(mean_vector[0], mean_vector[1],'o', markersize=5, color='green', alpha=0.6)

green_patch = mpatches.Patch(color='green', label='Mean')
plt.legend(handles=[green_patch])

red_patch = mpatches.Patch(color='red', label='Principal Eigenvectors')
plt.legend(handles=[red_patch])

blue_patch = mpatches.Patch(color='blue', label='Murder Data')
plt.legend(handles=[blue_patch, red_patch, green_patch])


# Part C
# 
# 1. Plot vairance (eigenvalues) against PC index
# 
# 2. Plot cumulative variance against the number of PCs used.
# 
# 3. Report how many PCs are needed to capture 90% and 95% variance in the dataset

# In[6]:

import pandas as pd 
evals, evecs = pca(XTrain)

print(len(evals))
print(len(evecs))

#print(pd.Series(evals))

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
#plt.axis('equal')

plt.title('Variance versus Principal Component Index')
plt.xlabel('Principal Components Index in Descending Order')
plt.ylabel('Projected Variance (Eigenvalues)')

# set the labels of the xticks
xticks( arange(13), ('PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13') )

plt.plot(evals)
plt.grid()             
plt.show();


# In[6]:

#run PCA on XTrain
evals, evecs = pca(XTrain)

#Create plot
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)

plt.title('Cumulative Variance over Number of Used Principal Components')
plt.xlabel('Number of used Principal Components')
plt.ylabel('Cumulative Normalized Variance (Eigenvalues)')

# set the labels of the xticks
xticks( arange(13), ('PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13') )

#cumulative
#np.cumsum documentation: https://docs.scipy.org/doc/numpy/reference/generated/numpy.cumsum.html
c_var = np.cumsum(evals/np.sum(evals))
plt.plot(c_var)
plt.grid()
plt.show();


# In[7]:

#Find percentage of variance captured by first three PCs.

print("90% variance: " + str(c_var[0:2]))

print("95% variance: " + str(c_var[0:3]))


# Exercise 2
# 
# 1) Project pesticide data points onto the first 2 PC of the dataset
# 
# 2) Produce a plot

# In[8]:

#Center data
mean_vector1 = [float(sum(n))/len(n) for n in zip(*XTrain)]

one = np.array(XTrain[:,0] - mean_vector1[0])
two = np.array(XTrain[:,1] - mean_vector1[1])
three = np.array(XTrain[:,2] - mean_vector1[2])
four = np.array(XTrain[:,3] - mean_vector1[3])
five = np.array(XTrain[:,4] - mean_vector1[4])
six = np.array(XTrain[:,5] - mean_vector1[5])
seven = np.array(XTrain[:,6] - mean_vector1[6])
eight = np.array(XTrain[:,7] - mean_vector1[7])
nine = np.array(XTrain[:,8] - mean_vector1[8])
ten = np.array(XTrain[:,9] - mean_vector1[9])
eleven = np.array(XTrain[:,10] - mean_vector1[10])
twelve = np.array(XTrain[:,11] - mean_vector1[11])
thirteen = np.array(XTrain[:,12] - mean_vector1[12])

data1 = np.column_stack((one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen))

#Perform PCA on centered data
evals, evecs = pca(data1)
matrix_w = np.hstack((evecs[0].reshape(13,1), evecs[1].reshape(13,1)))
#transform into new subspace
transformed = data1.dot(matrix_w)

#Plot data 
figure1 = plt.figure(figsize=(8,8))
ax1 = figure1.add_subplot(111)
ax1.scatter(transformed[:,0], transformed[:,1], marker = "x", color='orange', label='class1')

plt.title('Pesticide Data over Prinicpal Components')
plt.xlabel('Prinicpal Component 1')
plt.ylabel('Principal Component 2')

orange_patch = mpatches.Patch(color='orange', label='Pesticide Data')
plt.legend(handles=[orange_patch])


# Exerise 3
# 
# 1) Perform clustering using k-means algorithm. 
# 
# 2) Number of cluster centers is two (k = 2)

# In[9]:

#Inspiration from: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
#Inspiration from: https://pythonprogramming.net/k-means-from-scratch-machine-learning-tutorial/
#Inspiration from: https://www.datascience.com/blog/k-means-clustering

def clustering(data, k=2, tol=0.0001, max_iter=300): #default tolerance of scikit learn is 0.0001, max_iter is 300

    centroids = {}

    #Initialize the centroids 
    #the first k = 2 elements in the dataset will be the initial centroids, like asked 
    for i in range(k):
        centroids[i] = data[i]

    #Perform a single trial
    #Execute a singe iteration through features in data (via max_iter value) 
    for i in range(max_iter):
        classes = {}

        #Only looking to compare data points to first two centroids 
        #create two dict keys
        for i in range(k):
            classes[i] = []
        
        #iterate through our features
        for feature in data:
            
            #Euclidean Distance 
            #find the distance between the query point (features) and centroids; choose the nearest centroid
            distances = [np.linalg.norm(feature-centroids[centroid]) for centroid in centroids]
            
            #append the cluster list within classes with the data point’s feature vector.
            classification = distances.index(min(distances))
            classes[classification].append(feature)

        #make function to re-calculate the cluster centroids
        #store value of centroids per iteration
        #Dictionary is necessary to store centriod values that the iterations reutrn
        last_centroids = dict(centroids)

        #Note to self: clustering is based on these centriods! 
        #cluster datapoints must be averaged to find the new centriods
        for classification in classes:
            centroids[classification] = np.average(classes[classification],axis=0)
        
        #necessary to have in order to break out of the loop
        optimized = True

        #compare new and previous centroids 
        for centroid in centroids:
            og_centroid = last_centroids[centroid]
            current_centroid = centroids[centroid]
            
            #disables the warning
            np.seterr(divide='ignore')
            #If the centroid is lower than the value of tolerance then break out of the iterations
            if np.sum((current_centroid-og_centroid)/og_centroid) > tol:
                optimized = False
        
        #Based on tolerance
        if optimized:
            break

    return(centroids)

#TRY WITH DATA 
clustering(XTrain)


# In[10]:

from sklearn.cluster import KMeans

starting_point = np.vstack((XTrain[0,], XTrain[1,]))

kmeans = KMeans(n_clusters=2, n_init=1, init=starting_point, max_iter=300, algorithm="full").fit(XTrain) 
print(kmeans.cluster_centers_) 


# In[ ]:



