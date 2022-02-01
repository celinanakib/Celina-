#!/usr/bin/env python
# coding: utf-8

# <div style="background-color: #c1f2a5">
#
#
#
# Here, we are going to implement an analysis along the lines of Shepard’s Universal Law of Generalization.
#

# </div>

# In[3]:


import numpy as np
import matplotlib.pyplot as plt


# ## Q1 Consequential Regions
#
# Imagine that you have some one-dimensional stimuli and a consequential region to which you generalize some property. Since the actual consequential region is unknown, we need to approximate it using a number of possible regions. We first need to construct 10,000 consequential regions by randomly (i.e., uniformly) selecting a start and an end points in the range of **[-10,10]**.  We will later assume that all of your consequential regions are equally likely.
#
# Visualize the first 100 consequential regions in your list by plotting them as 100 successive **vertical lines**. The $x$-axis is the index of each consequential region and the $y$-axis shows the given consequential region at index $i$, ranging from starti to endi. Be sure to:
#
# In[4]:

n = 10000
CR_list = []

#first construct 10,000 consequential regions by randomly (i.e., uniformly)
#selecting a start and an end points in the range of [-10,10].
#Please use real numbers, NOT integers
#store these consequential regions in a list [ (start1, end1), (start2, end2), ... ].

def construct_regions(n):
    return [sorted((np.random.uniform(-10, 10), np.random.uniform(-10, 10))) for i in range(0, n) ]

#Visualize the first 100 consequential regions in your list by plotting them as 100 successive vertical lines.

figure, axis = plt.subplots()
CR_list = construct_regions(n)
first_100 = CR_list[:100]
x = np.arange(0.0,100.0,1.0)

ymin, ymax = [min(each) for each in first_100], [max(each) for each in first_100]
axis.vlines(x, ymin, ymax)

plt.title('First 100 consequential regions in list')
plt.ylabel('Consequential region at index i')
plt.xlabel('Index of each consequential region')
plt.xticks(np.arange(0.0,100.0,5.0))

figure.savefig('PS5_Q1.png')


# ## Q2 Helper Function
# Function called `contains` that takes a region and a point and checks if the region contains the given point.

# In[5]:

def contains(region, point):
    """ Takes a region (a 2-tuple) and a point (a float) and
    Returns:
      - 'True' if the point is contained in the region (boundaries defined by the 2-tuple)
      - 'False' if the point is outside of the region """
    if region[0] <= point and point <= region[1]:
        return True
    else:
        return False

#iterating through each of the 10,000 regions
# iterating through each of the 200 points created,
#checking (by contains function) if each point falls within each region
#counting how many times the contains function returns true

point = np.arange(-10.0,10.0,0.1)
proportions_point = []

for i in point:
    proportions_count = 0
    for CR in CR_list:
        if contains(CR, i):
            proportions_count += 1
    proportions_point.append(proportions_count/len(CR_list))

x = np.arange(-10.0,10.0,0.1)
figure, axis = plt.subplots(figsize=(10, 5))
axis.plot(x, proportions_point)
plt.title('Proportion of Regions That Contain Point')
plt.ylabel('Proportion')
plt.xlabel('Each Point x')

figure.savefig('PS5_Q2.png')

# ## Q3 Conditional Probability
#
# Below is a function that takes a list of possible consequential regions and a point $x$ and returns the probability of the actual consequential region (as approximated by the given list of regions) containing point $x$ conditioned on it containing point 0. For sanity check, this probability should be 1 for $x = 0$. The computation uses the probability rule P(A|B) = P(A $\wedge$ B) / P(B).

# In[6]:


def conditional_probability(CR_list, point):
    """ Takes a list of possible regions (a list) and a point (a float) and
    Returns the probability (a float between 0 and 1) that the point is in the actual region
    Approximated by the given list of consequential regions """

    # Keep track of the counts of:
    # - Regions that contain 0
    init0 = 0
    # - Regions that contain both x and 0
    init1 = 0
    # Keep track of the probabilities of:
    # - Sampling 0 from regions that contain 0
    init0_probability = 0
    # - Sampling x from regions that contain both x and 0
    init1_probability = 0

    # Loop through the list of possible regions
    for CR in CR_list:
        # Check if each region contains 0
        if contains(CR, 0):
            # How many regions contain 0? (Sum the counts)
            init0 += 1
            # How likely is it to sample 0 from each region? (Sum the probabilities)
            region_length = CR[1] - CR[0]
            init0_probability += (1 / region_length)
            # Check if given region ALSO contains x
            if contains(CR, point):
                # How many regions contain both x and 0? (Sum the counts)
                init1 += 1
                # How likely is it to sample x from each region? (Sum the probabilities)
                region_length = CR[1] - CR[0]
                init1_probability += (1 / region_length)

    # Probability of sampling 0 from actual consequential region
    try:
        ave_probability_0 = init0_probability / init0
    except ZeroDivisionError:
        ave_probability_0 = 0
    # Probability of sampling both x and 0 from actual consequential region
    try:
        ave_probability_1 = init1_probability / init1
    except ZeroDivisionError:
        ave_probability_1 = 0

    # Return conditional probability by applying the rule in the prompt
    try:
        cond_prob = ave_probability_1 / ave_probability_0
    except ZeroDivisionError:
        cond_prob = 0

    return cond_prob
# Check function value at x = 1
print(conditional_probability(CR_list, 1))

# In[7]:


figure, axis = plt.subplots(figsize=(10, 5))

conditional_probabilities = []
for i in np.arange(0,10.0,0.1):
    conditional_probabilities.append(conditional_probability(CR_list, i))

x = np.arange(0,10.0,0.1)

plt.plot(x,conditional_probabilities, color='orangered')
plt.xlabel('x-values')
plt.ylabel('Probability at value x')
plt.title('Probability of having x in the actual consequential region conditioned on the region containing x = 0')

figure.savefig('PS5_Q3.png')


# ## Q4 Check Exponential Decrease
#
# One way to check if the curve in Q3 has an exponential decrease is to plot a logarithmic $y$-axis and look for a straight line. Why can we check whether the curve is exponential by doing this? Please provide your answer in Gradescope.

# ## Q5.1 Logarithmic y-axis

# In[8]:

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,3))

x1 = np.arange(-5.0,5.0,0.1)

probabilities_left = []
for i in np.arange(-5.0,5.0,0.1):
    probabilities_left.append(conditional_probability(CR_list, i))


x2 = np.arange(-9.9,9.9,0.1)
probabilities_right = []
for i in np.arange(-9.9,9.9,0.1):
    probabilities_right.append(conditional_probability(CR_list, i))

axes[0].plot(x1, np.log(probabilities_left),color='orangered')
axes[0].set_xlabel('x-values')
axes[0].set_ylabel('Probability at value X in Log Scale')
axes[0].set_title('Subplot 1')
#axes[0].set_yscale('log')

axes[1].plot(x2, np.log(probabilities_right))
axes[1].set_xlabel('x-values')
axes[1].set_ylabel('Probability at value X in Log Scale')
axes[1].set_title('Subplot 2')
#axes[1].set_yscale('log')

plt.tight_layout()

fig.savefig('PS5_Q5.png')

# ## Q6 Limited Number of Regions
#
# In previous questions, we've been assuming that people implement Shepard’s Universal Law of Generalization perfectly and we have been trying to approximate their behavior using 10,000 possible consequential regions. However, people have limited resources:z
# Here is a Re-plot Q3 using only 10, 100, and 1000 consequential regions in the same plot

# In[9]:

# Create 3 lists of CR's
CR_list10 = construct_regions(10)
CR_list100 = construct_regions(100)
CR_list1000 = construct_regions(1000)

# Calculate corresponding conditional probabilities
conditional_probabilities1 = [conditional_probability(CR_list10, point) for point in np.arange(0, 10, 0.1)]
conditional_probabilities2 = [conditional_probability(CR_list100, point) for point in np.arange(0, 10, 0.1)]
conditional_probabilities3 = [conditional_probability(CR_list1000, point) for point in np.arange(0, 10, 0.1)]

# Plot on the same figure
figure, axis = plt.subplots(figsize=(10, 5))
axis.plot(np.arange(0, 10, 0.1), conditional_probabilities1, "r")
axis.plot(np.arange(0, 10, 0.1), conditional_probabilities2, "b")
axis.plot(np.arange(0, 10, 0.1), conditional_probabilities3, "g")

axis.legend(["10", "100", "1000"])
axis.set_xlabel('x-values')
axis.set_ylabel('Probability at value x')
axis.set_title('Probability of having x in the actual consequential region conditioned on the region containing x = 0')


figure.savefig('PS5_Q6.png')

# In[10]:

n = 10000
normal_CR_list = []

def construct_regions(n):
    return [sorted((np.random.normal(0,3), np.random.normal(0,3))) for i in range(0, n) ]


normal_CR_list = construct_regions(n)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,3))

x1 = np.arange(-5.0,5.0,0.1)

probabilities_left = []
for i in np.arange(-5.0,5.0,0.1):
    probabilities_left.append(conditional_probability(normal_CR_list, i))

x2 = np.arange(-9.9,9.9,0.1)
probabilities_right = []
for i in np.arange(-9.9,9.9,0.1):
    probabilities_right.append(conditional_probability(normal_CR_list, i))

axes[0].plot(x1, np.log(probabilities_left),color='orangered')
axes[0].set_xlabel('x-values')
axes[0].set_ylabel('Probability at value X in Log Scale')
axes[0].set_title('Subplot 1')
#axes[0].set_yscale('log')

axes[1].plot(x2, np.log(probabilities_right))
axes[1].set_xlabel('x-values')
axes[1].set_ylabel('Probability at value X in Log Scale')
axes[1].set_title('Subplot 2')
#axes[1].set_yscale('log')

plt.tight_layout()


fig.savefig('PS5_Q7.png')


# <div style="background-color: #c1f2a5">
     
# </div>
#
#
# <center>
#   <img src="https://www.dropbox.com/s/7s189m4dsvu5j65/instruction.png?dl=1" width="300"/>
# </center>
#
# <div style="background-color: #c1f2a5">
#
# - Submit the `.py` file you just created in Gradescope's PS5-code.
#
# </div>
#
#
#
#
# </div>
#
# </div>
#
