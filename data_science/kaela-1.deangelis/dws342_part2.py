
# coding: utf-8

# In[1]:

import numpy as np

get_ipython().magic(u'pylab inline')
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


# In[2]:

#import and read toy data

#read diatoms datset
data = np.loadtxt('pca_toydata.txt')


# Exercise 3

# PCA algorithm from past assignment 

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
    
    #eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
    #print("pairs: " + str(eig_pairs))
        
    return(eig_vals_for_return, eig_vecs_for_return)


# In[4]:

evals, evecs = pca(data)

matrix_w = np.hstack((evecs[0].reshape(4,1), evecs[1].reshape(4,1)))
#transform into new subspace
transformed = data.dot(matrix_w)

#Plot data 
figure1 = plt.figure(figsize=(8,8))
ax1 = figure1.add_subplot(111)
ax1.scatter(transformed[:,0], transformed[:,1], marker = "x", color='orange', label='class1')

plt.title('Toy Data over Prinicpal Components')
plt.xlabel('Prinicpal Component 1')
plt.ylabel('Principal Component 2')

orange_patch = mpatches.Patch(color='orange', label='Toy Data')
plt.legend(handles=[orange_patch])


# In[5]:

#Perform PCA on data

del_d = numpy.delete(data,(101), axis=0)
new_d = numpy.delete(del_d,(100),axis=0)

evals, evecs = pca(new_d)

matrix_w = np.hstack((evecs[0].reshape(4,1), evecs[1].reshape(4,1)))
#transform into new subspace
transformed = new_d.dot(matrix_w)

#Plot data 
figure1 = plt.figure(figsize=(8,8))
ax1 = figure1.add_subplot(111)
ax1.scatter(transformed[:,0], transformed[:,1], marker = "x", color='orange', label='class1')

plt.title('Toy Data over Prinicpal Components')
plt.xlabel('Prinicpal Component 1')
plt.ylabel('Principal Component 2')

orange_patch = mpatches.Patch(color='orange', label='Toy Data')
plt.legend(handles=[orange_patch])


# In[ ]:



