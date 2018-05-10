
# coding: utf-8

# In[1]:

import numpy as np
import math
import matplotlib.pyplot as plt
import pylab
import matplotlib.patches as mpatches


# In[2]:

#define the derivative of f(x)
def derivative(x):
    return (20.0*x) - ((np.exp(-x/2.0))/2.0)

#inclide f(x) to create y position for plot
def f(x):
    return np.exp(-x/2.0)+(10.0*(x**2))

#create gradient descent function 
def gradient_descent(learning_rate, num_values_to_return):
    
    x = 1 #starting point 
    
    x_steps = [] #update starting point to improve the guess 
    
    y_coord = [] #y 
    
    max_iter=10000
    
    x_values = []
    y_values = []
    
    y_pos = None
    last_y_position = None
    
    for number in range(max_iter):
        
        #derivative plays the role of the gradient 
        #Set the direction to move
        direction = -derivative(x)
        
        #update rule
        step = x + (learning_rate * direction)
        
        #add the newly calculated step to x 
        x_steps.append(step)
        
        #calculate the y pos based off the new x
        last_y_pos = y_pos
        y_pos = f(x)
        y_coord.append(y_pos)
        
        #need to get x,y coordinates within 3 and 10 iterations for plots
        if (number < num_values_to_return):
            y_values.append(y_pos)
            x_values.append(x)
        
        #print statement for last value
        if (y_pos == last_y_pos):
            print("Converged at " + str(x) + " with " + str(number) + " iterations")
            break;
        
        x = step
        
        #break if magnitude of gradient falls below 10e-10 or after x amount of iterations 
        if (np.abs(x) < 10**-10):
            print("At 10^-10 after " + str(number) + " iterations")
            break
        
    return x_values,y_values


# In[3]:

#Perform GD with the four learning rates and 3 iterations
x1, y1 = gradient_descent(0.1, 3)
x01, y01 = gradient_descent(0.01, 3)
x001, y001 = gradient_descent(0.001, 3)
x0001, y0001 = gradient_descent(0.0001, 3)


# In[4]:

#inspiration from https://glowingpython.blogspot.dk/2013/02/visualizing-tangent.html
#plot learning rate 0.1 at 3 iterations 
#items below help create plot capable of showing tangent line
plot_range = np.linspace(-2,2,150)
plot_values = f(plot_range)

#create tangent lines
a = x1[0]
tan = f(a) + derivative(a) * (plot_range-a)  #tangent 
b = x1[1]
tan1 = f(b) + derivative(b) * (plot_range-b)  #tangent 2
c = x1[2]
tan2 = f(c) + derivative(c) * (plot_range-c)  #tangent 3

#create plot
figure = plt.figure(figsize=(8,8))
ax = figure.add_subplot(111)
ax.scatter(x1,y1)
ax.plot(plot_range, plot_values, 'purple', plot_range, tan, '--r', plot_range, tan1, '--r', plot_range, tan2, '--r')#, tan, tan1, tan2, '--r')
#create title
plt.title('Gradient Descent: Learning Rate 0.1, 3 Iterations')

#create legend
ax.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
purple_patch = mpatches.Patch(color='purple', label='f(x)')
plt.legend(handles=[purple_patch])

blue_patch = mpatches.Patch(color='blue', label='Gradient steps')
plt.legend(handles=[blue_patch])

red_patch = mpatches.Patch(color='red', label='Tangent lines')
plt.legend(handles=[purple_patch, blue_patch,red_patch])

plt.show();

#plot learning rate 0.01 at 3 iterations 
plot_range = np.linspace(-2,2,25)
plot_values = f(plot_range)

#create tangent lines
a = x01[0]
tan = f(a) + derivative(a) * (plot_range-a)  #tangent 
b = x01[1]
tan1 = f(b) + derivative(b) * (plot_range-b)  #tangent 2
c = x01[2]
tan2 = f(c) + derivative(c) * (plot_range-c)  #tangent 3

#create plot
figure = plt.figure(figsize=(8,8))
ax = figure.add_subplot(111)
ax.scatter(x01,y01)
ax.plot(plot_range, plot_values, 'purple', plot_range, tan, '--r', plot_range, tan1, '--r', plot_range, tan2, '--r')
#create title
plt.title('Gradient Descent: Learning Rate 0.01, 3 Iterations')

#create legend
ax.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
purple_patch = mpatches.Patch(color='purple', label='f(x)')
plt.legend(handles=[purple_patch])

blue_patch = mpatches.Patch(color='blue', label='Gradient steps')
plt.legend(handles=[blue_patch])

red_patch = mpatches.Patch(color='red', label='Tangent lines')
plt.legend(handles=[purple_patch, blue_patch,red_patch])

plt.show();

#plot learning rate 0.001 at 3 iterations 
plot_range = np.linspace(-2,2,25)
plot_values = f(plot_range)

#create tangent lines
a = x001[0]
tan = f(a) + derivative(a) * (plot_range-a)  #tangent 
b = x001[1]
tan1 = f(b) + derivative(b) * (plot_range-b)  #tangent 2
c = x001[2]
tan2 = f(c) + derivative(c) * (plot_range-c)  #tangent 3

#create plot
figure = plt.figure(figsize=(8,8))
ax = figure.add_subplot(111)
ax.scatter(x001,y001)
ax.plot(plot_range, plot_values, 'purple', plot_range, tan, '--r', plot_range, tan1, '--r', plot_range, tan2, '--r')
#create title
plt.title('Gradient Descent: Learning Rate 0.001, 3 Iterations')

#create legend
ax.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
purple_patch = mpatches.Patch(color='purple', label='f(x)')
plt.legend(handles=[purple_patch])

blue_patch = mpatches.Patch(color='blue', label='Gradient steps')
plt.legend(handles=[blue_patch])

red_patch = mpatches.Patch(color='red', label='Tangent lines')
plt.legend(handles=[purple_patch, blue_patch,red_patch])

plt.show();

#plot learning rate 0.0001 at 3 iterations 
plot_range = np.linspace(-2,2,25)
plot_values = f(plot_range)

#create tangent lines
a = x0001[0]
tan = f(a) + derivative(a) * (plot_range-a)  #tangent1
b = x0001[1]
tan1 = f(b) + derivative(b) * (plot_range-b)  #tangent 2
c = x0001[2]
tan2 = f(c) + derivative(c) * (plot_range-c)  #tangent 3

#create plot
figure = plt.figure(figsize=(8,8))
ax = figure.add_subplot(111)
ax.scatter(x0001,y0001)
ax.plot(plot_range, plot_values, 'purple', plot_range, tan, '--r', plot_range, tan1, '--r', plot_range, tan2, '--r')
#create title
plt.title('Gradient Descent: Learning Rate 0.0001, 3 Iterations')

#create legend
ax.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
purple_patch = mpatches.Patch(color='purple', label='f(x)')
plt.legend(handles=[purple_patch])

blue_patch = mpatches.Patch(color='blue', label='Gradient steps')
plt.legend(handles=[blue_patch])

red_patch = mpatches.Patch(color='red', label='Tangent lines')
plt.legend(handles=[purple_patch, blue_patch,red_patch])

plt.show();


# In[5]:

#Perform GD with the four learning rates and 10 iterations
x1b, y1b = gradient_descent(0.1, 10)
x01b, y01b = gradient_descent(0.01, 10)
x001b, y001b = gradient_descent(0.001, 10)
x0001b, y0001b = gradient_descent(0.0001, 10)


# In[6]:

#plot learning rate 0.1 at 10 iterations 
plot_range = np.linspace(-2,2,25)
plot_values = f(plot_range)

figure = plt.figure(figsize=(8,8))
ax = figure.add_subplot(111)
ax.scatter(x1b,y1b)
ax.plot(plot_range,plot_values, "purple")

plt.title('Gradient Descent: Learning Rate 0.1, 10 Iterations')

#create legend
ax.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
purple_patch = mpatches.Patch(color='purple', label='f(x)')
plt.legend(handles=[purple_patch])

blue_patch = mpatches.Patch(color='blue', label='Gradient steps')
plt.legend(handles=[purple_patch, blue_patch])

plt.show();

#plot learning rate 0.01 at 10 iterations 
plot_range = np.linspace(-2,2,25)
plot_values = f(plot_range)

figure = plt.figure(figsize=(8,8))
ax = figure.add_subplot(111)
ax.scatter(x01b,y01b)
ax.plot(plot_range,plot_values, "purple")

plt.title('Gradient Descent: Learning Rate 0.01, 10 Iterations')

#create legend
ax.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
purple_patch = mpatches.Patch(color='purple', label='f(x)')
plt.legend(handles=[purple_patch])

blue_patch = mpatches.Patch(color='blue', label='Gradient steps')
plt.legend(handles=[purple_patch, blue_patch])

plt.show();

#plot learning rate 0.001 at 10 iterations 
plot_range = np.linspace(-2,2,25)
plot_values = f(plot_range)

figure = plt.figure(figsize=(8,8))
ax = figure.add_subplot(111)
ax.scatter(x001b,y001b)
ax.plot(plot_range,plot_values, "purple")

plt.title('Gradient Descent: Learning Rate 0.001, 10 Iterations')

#create legend
ax.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
purple_patch = mpatches.Patch(color='purple', label='f(x)')
plt.legend(handles=[purple_patch])

blue_patch = mpatches.Patch(color='blue', label='Gradient steps')
plt.legend(handles=[purple_patch, blue_patch])

plt.show();

#plot learning rate 0.0001 at 10 iterations 
plot_range = np.linspace(-2,2,25)
plot_values = f(plot_range)

figure = plt.figure(figsize=(8,8))
ax = figure.add_subplot(111)
ax.scatter(x0001b,y0001b)
ax.plot(plot_range,plot_values, "purple")

plt.title('Gradient Descent: Learning Rate 0.0001, 10 Iterations')

#create legend
ax.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
purple_patch = mpatches.Patch(color='purple', label='f(x)')
plt.legend(handles=[purple_patch])

blue_patch = mpatches.Patch(color='blue', label='Gradient steps')
plt.legend(handles=[purple_patch, blue_patch])

plt.show();


# In[ ]:



