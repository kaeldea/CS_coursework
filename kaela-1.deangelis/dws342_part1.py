
# coding: utf-8

# In[1]:

import numpy as np

get_ipython().magic(u'pylab inline')
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


# In[2]:

#read diatoms datset
data = np.loadtxt('diatoms.txt')

#pesticide dataset
dataTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
dataTest = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')

#split crop data into input variables and labels
XTrain = dataTrain[:, :-1] #13 dimensions without label
YTrain = dataTrain[:, -1] #only labels

XTest = dataTest[:, :-1] #13 dimensions without label
YTest = dataTest[:, -1] #only labels


# Exercise 1
# 
# 1. Plot of one cell

# In[3]:

#Separate data for first cell
x_coord = data[0][::2]
y_coord = data[0][1::2]

#stack to make clear coordinates
re_data = np.column_stack([x_coord, y_coord])

#stack first and last row to connect
first_last = np.column_stack([re_data[0], re_data[89]]).T

#create figure, plot data
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.axis('equal')

#plt.scatter(plot[:,0], plot[:,1], color='green')
plt.plot(re_data[:,0], re_data[:,1], color='green')
plt.plot(first_last[:,0], first_last[:,1], color='green')


plt.title('Shape of One Cell')
plt.xlabel('X Shape Features')
plt.ylabel('Y Shape Features')


# 2. Plot of all cells

# In[4]:

#Create function to arrange data into suitable x,y coordinates 
def xy_coord_arr(data):

    arranged_data = []
    
    for i in data:
        
        x_coord = i[::2]
        
        y_coord = i[1::2]
        
        all_arrays = np.column_stack([x_coord, y_coord])
        
        arranged_data.append(all_arrays)
    
    return(arranged_data)

arr_data = xy_coord_arr(data)

#Create figure and plot data
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.axis('equal')
    
plt.title('Shape of All Cells')
plt.xlabel('X Shape Features')
plt.ylabel('Y Shape Features')

#loop to plot all diatoms
for i in (arr_data):
    
    plt.plot(i[:,0], i[:,1])

#plot first and last cell to connect
for i in (arr_data):
  
    #stack first and last row to connect
    first_last = np.column_stack([i[0], i[89]]).T
    
    plt.plot(first_last[:,0], first_last[:,1])



# Exercise 2
# 
# Three parts

# PCA algorithm from past assignment 

# In[5]:

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


# Part A. Plot with sequences of cells showing the variance: PC1

# In[6]:

#PCA
evals, evecs = pca(data)

#mean_vector of all the data 
mean_vector = [float(sum(n))/len(n) for n in zip(*data)]

#standard deviation of eigenvalue for the first eigenvector
s1 = np.sqrt(evals[0])

#calculate the five 'cells
#Arrange data within cells to suitable x,y coordinates
#Connect first and last coordinates 

#First 
first_cell = (mean_vector - (2 * (s1 * evecs[0])))
x_coord = first_cell[::2]
y_coord = first_cell[1::2]

arr_data1 = np.column_stack([x_coord, y_coord])#This conenct first and last cell
fir_last1 = np.column_stack([arr_data1[89], arr_data1[0]]).T

#Second
second_cell = (mean_vector - (s1 * evecs[0]))
x_coord2 = second_cell[::2]
y_coord2 = second_cell[1::2]

arr_data2 = np.column_stack([x_coord2, y_coord2])#This conenct first and last cell
fir_last2 = np.column_stack([arr_data2[89], arr_data2[0]]).T

#Third
third_cell = (mean_vector)
x_coord3 = third_cell[::2]
y_coord3 = third_cell[1::2]

arr_data3 = np.column_stack([x_coord3, y_coord3])#This conenct first and last cell
fir_last3 = np.column_stack([arr_data3[89], arr_data3[0]]).T

#Fourth
fourth_cell =  (mean_vector + (s1 * evecs[0]))
x_coord4 = fourth_cell[::2]
y_coord4 = fourth_cell[1::2]

arr_data4 = np.column_stack([x_coord4, y_coord4])#This conenct first and last cell
fir_last4 = np.column_stack([arr_data4[89], arr_data4[0]]).T

#Fifth
fifth_cell = (mean_vector + (2 * (s1 * evecs[0])))
x_coord5 = fifth_cell[::2]
y_coord5 = fifth_cell[1::2]

arr_data5 = np.column_stack([x_coord5, y_coord5])#This conenct first and last cell
fir_last5 = np.column_stack([arr_data5[89], arr_data5[0]]).T

#Create Plot
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.axis('equal')

plt.title('Spatial Variance in First Prinipal Component')
plt.xlabel('X Shape Features')
plt.ylabel('Y Shape Features')

#color shading
reds = plt.get_cmap('Reds')

#Plot first cell
plt.plot(x_coord, y_coord, color = reds(0.2))
plt.plot(fir_last1[:,0], fir_last1[:,1], color = reds(0.2))

#Plot second cell
plt.plot(x_coord2, y_coord2, color= reds(0.4))
plt.plot(fir_last2[:,0], fir_last2[:,1], color = reds(0.4))

#Plot third cell
plt.plot(x_coord3, y_coord3, color= reds(0.6))
plt.plot(fir_last3[:,0], fir_last3[:,1], color = reds(0.6))

#Plot fourth cell
plt.plot(x_coord4, y_coord4, color = reds(0.8))
plt.plot(fir_last4[:,0], fir_last4[:,1], color = reds(0.8))

#Plot fifth cell
plt.plot(x_coord5, y_coord5, color = reds(0.9))
plt.plot(fir_last5[:,0], fir_last5[:,1], color = reds(0.9))

#Create legend
L_red_patch = mpatches.Patch(color=reds(0.2), label='m - 2(sigma1*eigenvector1)')
plt.legend(handles=[L_red_patch])

L2_red_patch = mpatches.Patch(color=reds(0.4), label='m - (sigma1*eigenvector1)')
plt.legend(handles=[L2_red_patch])

L3_red_patch = mpatches.Patch(color=reds(0.6), label='mean')
plt.legend(handles=[L3_red_patch])

D1_red_patch = mpatches.Patch(color=reds(0.8), label='m + (sigma1*eigenvector1)')
plt.legend(handles=[D1_red_patch])

D2_red_patch = mpatches.Patch(color=reds(0.9), label='m + 2(sigma1*eigenvector1)')
plt.legend(handles=[L_red_patch, L2_red_patch, L3_red_patch, D1_red_patch, D2_red_patch])


# Part B. Plot with sequences of cells showing the variance: PC2

# In[7]:

#PCA
evals, evecs = pca(data)

#mean_vector of all the data 
mean_vector = [float(sum(n))/len(n) for n in zip(*data)]

#standard deviation of eigenvalue for the second eigenvector
s2 = np.sqrt(evals[2])

#calculate the five 'cells'
#Arrange data within cells to suitable x,y coordinates
#Connect first and last cell

#First
first_cell = (mean_vector - (2 * (s2 * evecs[1])))
x_coord20 = first_cell[::2]
y_coord20 = first_cell[1::2]

arr_data20 = np.column_stack([x_coord20, y_coord20]) #This conenct first and last cell
fir_last20 = np.column_stack([arr_data20[89], arr_data20[0]]).T


#Second
second_cell = (mean_vector - (s2 * evecs[1]))
x_coord22 = second_cell[::2]
y_coord22 = second_cell[1::2]

arr_data22 = np.column_stack([x_coord22, y_coord22])#This conenct first and last cell
fir_last22 = np.column_stack([arr_data22[89], arr_data22[0]]).T

#Third
third_cell = (mean_vector)
x_coord23 = third_cell[::2]
y_coord23 = third_cell[1::2]

arr_data23 = np.column_stack([x_coord23, y_coord23])#This conenct first and last cell
fir_last23 = np.column_stack([arr_data23[89], arr_data23[0]]).T

#Fourth
fourth_cell = (mean_vector + (s2 * evecs[1]))
x_coord24 = fourth_cell[::2]
y_coord24 = fourth_cell[1::2]

arr_data24 = np.column_stack([x_coord24, y_coord24])#This conenct first and last cell
fir_last24 = np.column_stack([arr_data24[89], arr_data24[0]]).T

#Fifth
fifth_cell = (mean_vector + (2 * (s2 * evecs[1])))
x_coord25 = fifth_cell[::2]
y_coord25 = fifth_cell[1::2]

arr_data25 = np.column_stack([x_coord25, y_coord25])#This conenct first and last cell
fir_last25 = np.column_stack([arr_data25[89], arr_data25[0]]).T

#Make plot
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.axis('equal')
    
plt.title('Spatial Variance Over Second Prinipal Component')
plt.xlabel('X Shape Features')
plt.ylabel('Y Shape Features')

#Color shading
reds = plt.get_cmap('Reds')

#Plot first cell
plt.plot(x_coord20, y_coord20, color = reds(0.2))
plt.plot(fir_last20[:,0], fir_last20[:,1], color = reds(0.2))

#Plot second cell
plt.plot(x_coord22, y_coord22, color = reds(0.4))
plt.plot(fir_last22[:,0], fir_last22[:,1], color = reds(0.4))

#Plot third Cell
plt.plot(x_coord23, y_coord23, color = reds(0.6))
plt.plot(fir_last23[:,0], fir_last23[:,1], color = reds(0.6))

#Plot fourth cell
plt.plot(x_coord24, y_coord24, color = reds(0.8))
plt.plot(fir_last24[:,0], fir_last24[:,1], color = reds(0.8))

#Plot fifth cell
plt.plot(x_coord25, y_coord25, color = reds(0.9))
plt.plot(fir_last25[:,0], fir_last25[:,1], color = reds(0.9))

#Create legend
L_red_patch = mpatches.Patch(color=reds(0.2), label='m - 2(sigma2*eigenvector2)')
plt.legend(handles=[L_red_patch])

L2_red_patch = mpatches.Patch(color=reds(0.4), label='m - (sigma2*eigenvector2)')
plt.legend(handles=[L2_red_patch])

L3_red_patch = mpatches.Patch(color=reds(0.6), label='mean')
plt.legend(handles=[L3_red_patch])

D1_red_patch = mpatches.Patch(color=reds(0.8), label='m + (sigma2*eigenvector2)')
plt.legend(handles=[D1_red_patch])

D2_red_patch = mpatches.Patch(color=reds(0.9), label='m + 2(sigma2*eigenvector2)')
plt.legend(handles=[L_red_patch, L2_red_patch, L3_red_patch, D1_red_patch, D2_red_patch])


# Part C: Plot with sequences of cells showing the variance: PC3

# In[8]:

#PCA
evals, evecs = pca(data)

#mean_vector of all the data 
mean_vector = [float(sum(n))/len(n) for n in zip(*data)]

#standard deviation of eigenvalue for the third eigenvector
s3 = np.sqrt(evals[2])

#calculate the five 'cells'
#Arrange data within cells to suitable x,y coordinates
#Connect first and last cell cooridnates 

#First 
first_cell = (mean_vector - (2 * (s3 * evecs[2])))
x_coord30 = first_cell[::2]
y_coord30 = first_cell[1::2]

arr_data30 = np.column_stack([x_coord30, y_coord30])#This conenct first and last cell
fir_last30 = np.column_stack([arr_data30[89], arr_data30[0]]).T

#Second
second_cell = (mean_vector - (s3 * evecs[2]))
x_coord32 = second_cell[::2]
y_coord32 = second_cell[1::2]

arr_data32 = np.column_stack([x_coord32, y_coord32])#This conenct first and last cell
fir_last32 = np.column_stack([arr_data32[89], arr_data32[0]]).T

#Third
third_cell = (mean_vector)
x_coord33 = third_cell[::2]
y_coord33 = third_cell[1::2]

arr_data33 = np.column_stack([x_coord33, y_coord33])#This conenct first and last cell
fir_last33 = np.column_stack([arr_data33[89], arr_data33[0]]).T

#Fourth
fourth_cell = (mean_vector + (s3 * evecs[2]))
x_coord34 = fourth_cell[::2]
y_coord34 = fourth_cell[1::2]

arr_data34 = np.column_stack([x_coord34, y_coord34])#This conenct first and last cell
fir_last34 = np.column_stack([arr_data34[89], arr_data34[0]]).T

#Fifth
fifth_cell = (mean_vector + (2 * (s3 * evecs[2])))
x_coord35 = fifth_cell[::2]
y_coord35 = fifth_cell[1::2]

arr_data35 = np.column_stack([x_coord35, y_coord35])#This conenct first and last cell
fir_last35 = np.column_stack([arr_data35[89], arr_data35[0]]).T

#merge the five cells to plot
#cell_data = vstack((first_cell,second_cell, third_cell, fourth_cell, fifth_cell))

#Create Plot
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.axis('equal')
    
plt.title('Spatial Variance Over Third Prinipal Component')
plt.xlabel('X Shape Features')
plt.ylabel('Y Shape Features')

#color shading
reds = plt.get_cmap('Reds')

#Plot first 
plt.plot(x_coord30, y_coord30, color = reds(0.2))
plt.plot(fir_last30[:,0], fir_last30[:,1], color = reds(0.2))

#Plot second
plt.plot(x_coord32, y_coord32, color = reds(0.4))
plt.plot(fir_last32[:,0], fir_last32[:,1], color = reds(0.4))

#Plot third
plt.plot(x_coord33, y_coord33, color = reds(0.6))
plt.plot(fir_last33[:,0], fir_last33[:,1], color = reds(0.6))

#Plot fourth
plt.plot(x_coord34, y_coord34, color = reds(0.8))
plt.plot(fir_last34[:,0], fir_last34[:,1], color = reds(0.8))

#Plot fifth
plt.plot(x_coord35, y_coord35, color = reds(0.9))
plt.plot(fir_last35[:,0], fir_last35[:,1], color = reds(0.9))

#Create legend
L_red_patch = mpatches.Patch(color=reds(0.2), label='m - 2(sigma3*eigenvector3)')
plt.legend(handles=[L_red_patch])

L2_red_patch = mpatches.Patch(color=reds(0.4), label='m - (sigma3*eigenvector3)')
plt.legend(handles=[L2_red_patch])

L3_red_patch = mpatches.Patch(color=reds(0.6), label='mean')
plt.legend(handles=[L3_red_patch])

D1_red_patch = mpatches.Patch(color=reds(0.8), label='m + (sigma3*eigenvector3)')
plt.legend(handles=[D1_red_patch])

D2_red_patch = mpatches.Patch(color=reds(0.9), label='m + 2(sigma3*eigenvector3)')
plt.legend(handles=[L_red_patch, L2_red_patch, L3_red_patch, D1_red_patch, D2_red_patch])


# In[ ]:



