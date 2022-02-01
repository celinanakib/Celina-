#!/usr/bin/env python
# coding: utf-8

# <div style="background-color: #c1f2a5">
#
# #
# Here we will implement multidimensional scaling (MDS) from scratch.  MDS attempts to find an arrangement of points such that the distances between points match human-judged similarities.
#

# </div>

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# to import the data set
from scipy.io import loadmat


# ## Import and examine data
#
# We will be using a data set from Romney, A. K., Brewer, D. D., & Batchelder, W. H. (1993). Predicting Clustering from Semantic Structure. Psychological Science, 4(1), 28-34, via https://faculty.sites.uci.edu/mdlee/similarity-data/. The data set is saved in PS4_dataset.mat, and includes pairwise similarity measures between 21 sports.
# As our first step, we will download and examine the data:
#

# In[2]:

data_set = loadmat('PS4_dataset.mat')
similarity = data_set['similarity']
sport_names = data_set['sport_names']


# As we can see, our data contains information for 21 different sports as listed below:

# In[3]:


print(sport_names)


# We also have a similarity matrix for each sport, which gives us the psychological similarity of that sport with all the other sports in the data:

# In[4]:


#Look at the first similarity matrix, which corresponds to football's similarity with itself and all other sports
print(similarity[0])


# ## Q1. Visualize similarity

# Plot the "similarity" measures from the data as a heatmap.

# In[27]:

# Import and create heatmap

figure, ax = plt.subplots()
im = ax.imshow(similarity)

ax.set_xticks(np.arange(len(sport_names)))
ax.set_yticks(np.arange(len(sport_names)))

# Rotate the labels and set their alignment.
ax.set_xticklabels(sport_names, rotation = 45)
ax.set_yticklabels(sport_names, rotation = 45)


# Title and Plot
ax.set_title("Heatmap of sport similarity")
plt.colorbar(im)
plt.show()


figure.savefig('PS4_Q1.png')


# ## Q2. Distance
#
# To implement MDS, we need a measure of psychological **distance**. The dataset includes measures of similarity, not distance.
#
# Here we will use *d = 1-s* as a  method to transform similarity to distance.
#
# The function will convert all similarity measures in the dataset into distances, using the above provided transformation method. Function should return the output called distance
#
# Plot a scatterplot of the dataset's distances (x axis) against their similarity (y axis).
#

# In[28]:

figure, axis = plt.subplots()

def psychological_distance(similarity):
    global distance
    distance = 1 - similarity
    return distance


axis.set_xlabel("Distance")
axis.set_ylabel("Similarity")
axis.set_title("Scatter Plot Depicting Distances against their similarity")
plt.plot(psychological_distance(similarity), similarity,'o', color='black')

figure.savefig('PS4_Q2.png')


# ## Q3. Stress
#
# To perform MDS, we will try to find, for each sport i, a position $p_i=(x_i,y_i)$ in the 2d space that captures the participants’ similarities. To do so, we will build an algorithm that minimizes the stress. We’ll define stress slightly differently than in class- the squared difference between psychological distance $\psi_{ij}= (1-s_{ij})$ and the MDS distance in 2D space:
#
# $$ \mathrm{Stress \ S} = \sum_{i > j} (\psi_{ij} - dist(p_i,p_j))^2$$
#
# Where $\psi$ is the psychological distance between sport i and sport j that was reported by subjects, and *dist(pi,pj)* corresponds to the **Euclidean distance**:$\sqrt{(x_i-x_j)^2 + (y_i-y_j)^2}$
#
# Function will compute the Euclidean Distance between two points $p_1$ and $p2$. Then, another fucntion will takes a $(n,2)$ (n=number of sports) matrix of $(x,y)$ positions for each sport, and computes the stress based on the equation above, using your Euclidean Distance function.
#

# In[29]:


def EuclideanDistance(p1,p2):
    ''' Takes positions defined by p1 and p2, and returns a euclidean distance value (single number).
    Implement EQ equation provided in the question. Hint: if p1=p2, the function should return the value of 0'''
    ED = np.linalg.norm(p1-p2)
    return ED


# In[30]:


def StressCalc(positions, distance):
    ''' Takes positions (n,2) and (n,n) matrix of distance measures
    (You will use the distance matrix from Q2).
    Uses these distances and the EuclideanDistance function above which computes ED based on positions
    to calculate the Stress between psychological and ED distances, according to the provided formula.'''
    stress = []
    for i in range(len(similarity[0])):
        for j in range(i, len(similarity[1])):
            stress.append((np.sum(distance[i][j]) - EuclideanDistance(positions[i],positions[j])) **2)
    return np.sum(stress)


# In[31]:


# Test case!
'''
Test case for StressCalc: create an array of positions, where each entry is 1.
Use this positions matrix and distance matrix from Q2 to call StressCalc function
'''

positions = np.ones((len(similarity),2))
print(['Stress value should be 111.57. Output stress value is: ' + str(StressCalc(positions,distance))])


# ## Q4. Gradient
# To minimize the stress, we will numerically compute the gradient using a multidimensional version of the simple rule for derivatives:
#
# $$ \frac{df}{dp}(p) = \frac{f(p+\delta)-f(p-\delta)}{2\delta}$$
#
# where $\delta$ takes on a small value, and $f$ is the stress function you wrote in the previous question. To compute the gradient, we will compute this approximate derivative with respect to each coordinate of each point.
#
# function will take an n-by-2 matrix (n=number of sports) of (x,y) positions for each sport and computes the gradient (i.e. applies the numerical rule above to each coordinate location). This should return an n-by-2 gradient matrix.
#
#
# Use $\delta$ = 0.01
#

# In[32]:


delta = .01
positions = np.random.rand(len(similarity),2)

gradient_output = np.empty(np.shape(positions))

def gradientDescent(delta, positions):
    for i in range(len(positions)):
        for j in range(len(positions[0])):
            positions_plus_delta= np.copy(positions)
            positions_minus_delta= np.copy(positions)
            positions_plus_delta[i,j] += delta
            positions_minus_delta[i,j] -= delta
            gradient_output[i,j] = ((StressCalc(positions_plus_delta, distance) - StressCalc(positions_minus_delta, distance))/(2*delta))
    return gradient_output

print(gradientDescent(delta,positions))

# In[33]:

p = np.empty([21,2])
p[:,0] = np.arange(0,21*0.04,0.04)
p[:,1] = np.arange(0,21*0.04,0.04)
print(gradientDescent(delta, p))


# ## Q5.1 MDS
#
# Write the MDS code: the code that follows a gradient in order to find positions that minimize the stress. Start from a random position, and be sure to take small steps in the direction of the gradient (e.g.  α*gradient, with step size  α=0.01), to find a set of positions that minimizes the stress. Use 100 steps of gradient descent.
#

# In[39]:

delta = .01
alpha = 0.01
stored_stress = []
current_position = np.random.rand(len(similarity),2)

def MDS(position, steps = 100):
    current_position = np.random.rand(len(similarity),2)
    for i in range(steps):
        current_position -= alpha *  gradientDescent(delta, current_position)
        stored_stress.append(StressCalc(current_position,distance))
    return current_position

MDS(current_position, steps = 100)


# ## Q5.2
#
# Plot the names of sports at the resulting coordinates.
# In[40]:

x1, y1 = [each[0] for each in current_position], [each[1] for each in current_position]
fig1, ax1 = plt.subplots()
ax1.scatter(x1, y1)
for i, txt in enumerate(sport_names):
    ax1.annotate(txt, (x1[i], y1[i]))

plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Sport Names at Resulting Coordinates')

fig1.savefig('PS4_Q5_2.png')


# ## Q5.3
# Plot the stress as a function of step number (x axis = step number, y axis= stress).
# In[41]:

figure, axis = plt.subplots()
steps = 100
x = np.arange(steps)
y = stored_stress
plt.plot(x, y)
plt.xlabel('Step Number')
plt.ylabel('Stress')
plt.title('Stress as a Function of Step Number')

figure.savefig('PS4_Q5_3.png')


# ## Q6. Validation
#
# Make a scatter plot of the distances obtained by running your MDS function vs. people's reported distances *d=(1-s)*.
#

# In[42]:


x = np.array([i[0] for i in current_position])
y = np.array([i[1] for i in current_position])
MDS_distances = np.sqrt(np.square(x - x.reshape(-1,1)) + np.square(y - y.reshape(-1,1)))


figure, axis = plt.subplots()
axis.scatter(psychological_distance(similarity), MDS_distances)
plt.xlabel('Psychological Distance')
plt.ylabel('MDS Distance')
plt.title('Scatter Plot of Distances: MDS Vs. Psychological Distance')


figure.savefig('PS4_Q6.png')


# ## Q7.1 Iterating MDS
#
# In[22]:

figure, ax = plt.subplots(3,3, sharex = True, sharey= True, figsize=(15,15))
updated_stress_values = []

for i in range(9):
    for i, ax in enumerate(ax.flatten()):
        coordinates = MDS(np.random.rand(len(similarity),2))
        updated_stress_values.append(stored_stress[-1])
        x, y = [each[0] for each in coordinates], [each[1] for each in coordinates]
        ax.scatter(x,y)
        ax.set_title(i)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        for j, txt in enumerate(sport_names):
            ax.annotate(txt, (x[j], y[j]))


# ## Q7.2 Best representation

# In[23]:

figure,axis = plt.subplots()
x = np.arange(9)
axis.scatter(x, updated_stress_values)
axis.set_xlabel('Iteration')
axis.set_ylabel('Final Stress Value of each iteration')
axis.set_title("Final stress value as a function of the MDS iteration")


figure.savefig('PS4_Q7_2.png')


# ## Q8
# Run MDS 2 times, with 2 different step sizes (α=.02 and  α=.05). Plot Stress over time for each run in the same plot. Don't forget to add a legend,labeling which MDS step size the line refers to, in addition to the usual axis labels and title. What happens if you use a bigger step in your MDS? Why?
#
# In[24]:

figure, ax = plt.subplots()

stored_stress = []
alpha = 0.02
MDS1 = MDS(np.random.rand(len(similarity),2))
ax.plot(np.arange(100), stored_stress, label='alpha = 0.02')

stored_stress = []
alpha = 0.05
MDS2 = MDS(np.random.rand(len(similarity),2))
ax.plot(np.arange(100), stored_stress, label='alpha = 0.05')


plt.xlabel('Step Number')
plt.ylabel('Stress')
plt.title('Plot of Stress vs. Step Number')
ax.legend()

figure.savefig('PS4_Q8.png')


# <div style="background-color: #c1f2a5">
#
# </div>

# In[ ]:
