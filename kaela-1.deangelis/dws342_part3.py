
# coding: utf-8

# Exercise 4

# In[1]:

import numpy as np
from sklearn.cluster import KMeans

get_ipython().magic(u'pylab inline')
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


# In[2]:

#pesticide dataset
dataTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
dataTest = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')

#split crop data into input variables and labels
XTrain = dataTrain[:, :-1] #13 dimensions without label
YTrain = dataTrain[:, -1] #only labels

XTest = dataTest[:, :-1] #13 dimensions without label
YTest = dataTest[:, -1] #only labels


# 1. Use PCA algorithm from Assignment 3

# In[3]:

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


# 3. Implement PCA and KMeans Clustering on XTrain data
# 
# 4. Create plot projecting labelled XTrain data onto first two principal components
# 
# 5. Add to plot the centroids from applying the clustering algorithm to XTrain. 

# In[4]:

#Inspiration from: https://plot.ly/ipython-notebooks/principal-component-analysis/

#Perform PCA on centered data
evals, evecs = pca(XTrain)
matrix_w = np.hstack((evecs[0].reshape(13,1), evecs[1].reshape(13,1)))

#transform into new subspace
transformed = XTrain.dot(matrix_w)

#Plot data 
figure1 = plt.figure(figsize=(10,10))
ax1 = figure1.add_subplot(111)
ax1.axis('equal')

#Create variable for coloring labels
y = YTrain
ax1.scatter(transformed[:,0], transformed[:,1], marker = "x", c=y, cmap = cm.Set3)

#Perform KMeans and add centroids to plot 
starting_point = np.vstack((XTrain[0,], XTrain[1,]))
kmeans = KMeans(n_clusters=2, n_init=1, init=starting_point, max_iter=300, algorithm="full").fit(XTrain) 
centers = (kmeans.cluster_centers_)
transformed_cent = centers.dot(matrix_w)

for centroid in transformed_cent:
    ax1.scatter(centroid[0], centroid[1], marker="o", color="k", s=150, linewidths=5)

#Title, axis labels and legend
plt.title('Pesticide Data over Prinicpal Components')
plt.xlabel('Prinicpal Component 1')
plt.ylabel('Principal Component 2')

black_patch = mpatches.Patch(color='black', label='Cluster Centers')
plt.legend(handles=[black_patch])

yellow_patch = mpatches.Patch(color='yellow', label='Pesticide Data: Class 1')
plt.legend(handles=[yellow_patch])

blue_patch = mpatches.Patch(color='lightblue', label='Pesticide Data: Class 0')
plt.legend(handles=[black_patch, blue_patch, yellow_patch])


# In[ ]:



