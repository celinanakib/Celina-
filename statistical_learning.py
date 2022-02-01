#!/usr/bin/env python
# coding: utf-8

# <div style="background-color: #c1f2a5">
#
#
#
# Here, we will play the number game (Tenenbaum, 2000). We have a mysterious device that accepts some numbers between 1 and 100 but not others. We will use Bayes' rule to figure out what rule our device might follow and which numbers are likely to be accepted, given some data about the numbers that the device has accepted.
#
# </div>

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# ## Introduction
# Consider the following hypotheses for all integers from 1 to 100:
# - H0: Even numbers from 2 to 100 (2, 4, 6, ..., 100)
# - H1: Odd numbers from 1 to 99 (1, 3, 5, ..., 99)
# - H2: Square numbers (1, 4, 9, ..., 100)
# - H3: Prime numbers (2, 3, 5, ..., 97)
# - H4: Multiples of 5 (5, 10, 15, ..., 100)
# - H5: Multiples of 10 (10, 20, 30, ..., 100)
# - H6: Powers of 2 (2, 4, 8, ..., 64)
# - H7: All numbers (1, 2, 3, ..., 100)
#
# Unless otherwise specified (in Q5), each hypothesis is assumed to be **equally likely** (P(H0) = ... = P(H7) = 1/8) and can be represented as a NumPy array containing acceptable numbers.

# ## Q1. Data
#
# To begin, let's create a NumPy array for each of our 8 hypotheses and store the 8 arrays in a list.
#
# Then in 8 subplots (1 per hypothesis), plot whether or not the hypothesis has each number from 1 to 100 (x-axis) in it (1 if yes, 0 if not, y-axis). You **MUST** follow the plotting instructions below to get full credit:
#

# In[2]:

def is_prime(n):
    """Determine if given number if prime"""
    status = True
    if n < 2:
        status = False
    else:
        for i in range(2, n):
            if n % i == 0:
                status = False
    return status

# In[3]:


# Create 8 hypotheses
H0 = np.arange(2, 101, 2)  # Even numbers from 2 to 100
H1 = np.arange(1, 100, 2)  # Odd numbers from 1 to 00
H2 = np.arange(1, 11) ** 2  # Square numbers
H3 = np.array([i for i in range(101) if is_prime(i)])  # Prime numbers
H4 = np.arange(5, 101, 5)  # Multiples of 5
H5 = np.arange(10, 101, 10)  # Multiples of 10
H6 = 2 ** np.arange(1, 7)  # Powers of 2
H7 = np.arange(1, 101)  # All numbers
H = [H0, H1, H2, H3, H4, H5, H6, H7]

# In[4]:

figure, axes = plt.subplots(4, 2, sharex='col', sharey='row', figsize=(16, 10))

x = np.arange(1,101)

def plot(H):
    y = []
    for i in x:
        if i in H:
            y.append(1)
        if i not in H :
            y.append(0)
    return y


axes[0][0].bar(x,plot(H0))
axes[0][0].set_title('H0')
axes[0][0].set_xlabel('x-value')
axes[0][0].set_ylabel('y-value (1 or 0)')

axes[0][1].bar(x,plot(H1))
axes[0][1].set_title('H1')
axes[0][1].set_xlabel('x-value')
axes[0][1].set_ylabel('y-value (1 or 0)')

axes[1][0].bar(x,plot(H2))
axes[1][0].set_title('H2')
axes[1][0].set_xlabel('x-value')
axes[1][0].set_ylabel('y-value (1 or 0)')

axes[1][1].bar(x,plot(H3))
axes[1][1].set_title('H3')
axes[1][1].set_xlabel('x-value')
axes[1][1].set_ylabel('y-value (1 or 0)')

axes[2][0].bar(x,plot(H4))
axes[2][0].set_title('H4')
axes[2][0].set_xlabel('x-value')
axes[2][0].set_ylabel('y-value (1 or 0)')

axes[2][1].bar(x,plot(H5))
axes[2][1].set_title('H5')
axes[2][1].set_xlabel('x-value')
axes[2][1].set_ylabel('y-value (1 or 0)')

axes[3][0].bar(x,plot(H6))
axes[3][0].set_title('H6')
axes[3][0].set_xlabel('x-value')
axes[3][0].set_ylabel('y-value (1 or 0)')

axes[3][1].bar(x,plot(H7))
axes[3][1].set_title('H7')
axes[3][1].set_xlabel('x-value')
axes[3][1].set_ylabel('y-value (1 or 0)')

figure.suptitle('Hypothesis Plots')

plt.tight_layout()

figure.savefig("PS8_Q1.png")


# ## Q2. Likelihood
#
# Here we will be computing the likelihood of data under a given hypothesis. Your data will be represented as an array (e.g., [3, 19, 63]) with a function called `compute_likelihood` that takes a NumPy array (each number is a **unique** integer between 1 and 100) and a hypothesis (H0-H7, as specified in Q1) as inputs and returns 1) an array containing the likelihood of each number independently (the likelihoods of 3, 19, and 63 separately) as well as 2) the likelihood of the entire array (the likelihood of [3, 19, 63]).
#

figure, axes = plt.subplots(4, 2, sharex='col', sharey='row', figsize=(16, 10))

x = np.arange(1,101)

def compute_likelihood(x,H):
    individual_likelihood = []
    array_likelihood = []
    for i in x:
        if i in H:
            likelihood = 1/len(H)
            individual_likelihood.append(likelihood)
        else:
            likelihood = 0
            individual_likelihood.append(likelihood)

    array_likelihood = np.prod(individual_likelihood)
    return individual_likelihood,array_likelihood


H0x,a = compute_likelihood(x, H0)
axes[0][0].bar(x,H0x)
axes[0][0].set_title('H0')
axes[0][0].set_xlabel('x-value')
axes[0][0].set_ylabel('likelihood')
axes[0][0].set_ylim([0,0.2])


H1x,a = compute_likelihood(x, H1)
axes[0][1].bar(x,H1x)
axes[0][1].set_title('H1')
axes[0][1].set_xlabel('x-value')
axes[0][1].set_ylabel('likelihood')
axes[0][1].set_ylim([0,0.2])

H2x,a = compute_likelihood(x, H2)
axes[1][0].bar(x,H2x)
axes[1][0].set_title('H2')
axes[1][0].set_xlabel('x-value')
axes[1][0].set_ylabel('likelihood')
axes[1][0].set_ylim([0,0.2])

H3x,a = compute_likelihood(x, H3)
axes[1][1].bar(x,H3x)
axes[1][1].set_title('H3')
axes[1][1].set_xlabel('x-value')
axes[1][1].set_ylabel('likelihood')
axes[1][1].set_ylim([0,0.2])

H4x,a = compute_likelihood(x, H4)
axes[2][0].bar(x,H4x)
axes[2][0].set_title('H4')
axes[2][0].set_xlabel('x-value')
axes[2][0].set_ylabel('likelihood')
axes[2][0].set_ylim([0,0.2])

H5x,a = compute_likelihood(x, H5)
axes[2][1].bar(x,H5x)
axes[2][1].set_title('H5')
axes[2][1].set_xlabel('x-value')
axes[2][1].set_ylabel('likelihood')
axes[2][1].set_ylim([0,0.2])

H6x,a = compute_likelihood(x, H6)
axes[3][0].bar(x,H6x)
axes[3][0].set_title('H6')
axes[3][0].set_xlabel('x-value')
axes[3][0].set_ylabel('likelihood')
axes[3][0].set_ylim([0,0.2])

H7x,a = compute_likelihood(x, H7)
axes[3][1].bar(x,H7x)
axes[3][1].set_title('H7')
axes[3][1].set_xlabel('x-value')
axes[3][1].set_ylabel('likelihood')
axes[3][1].set_ylim([0,0.2])


figure.suptitle("Likelihoods under a Hypothesis")

figure.tight_layout()
figure.savefig("PS8_Q2.png")


# ## Q3 - Bayes' rule
#
# Now let's calculate the posterior probability of each hypothesis in light of data. Below are 8 datasets you'll be working with:
#
# - (a) No data
# - (b) 50
# - (c) 53
# - (d) 50, 53
# - (e) 16
# - (f) 10, 20
# - (g) 2, 4, 8
# - (h) 2, 4, 8, 10
#
# Let's assume that each of our 8 hypotheses is equally likely prior to seeing data and that, under each hypothesis, each number allowed by it is also equally likely (the "size principle likelihood").
#
#  `Compute_posterior` will compute the posterior probability of each hypothesis given a dataset. Note that the 8 probabilities should sum to 1 then we will plot the posterior probability of each hypothesis.

# In[6]:


# Generate the 8 datasets
DataSets = []

D0 = np.empty(0)
DataSets.append(D0)

D1 = np.array([50])
DataSets.append(D1)

D2 = np.array([53])
DataSets.append(D2)

D3 = np.array([50, 53])
DataSets.append(D3)

D4 = np.array([16])
DataSets.append(D4)

D5 = np.array([10, 20])
DataSets.append(D5)

D6 = np.array([2, 4, 8])
DataSets.append(D6)

D7 = np.array([2, 4, 8, 10])
DataSets.append(D7)


# In[7]:

figure, axes = plt.subplots(4, 2, sharex='col', sharey='row', figsize=(16, 10))

H = [H0, H1, H2, H3, H4, H5, H6, H7]
priors = np.full(len(H),1/8)

def compute_posterior(dataset,hypothesis,priors):
    posterior_prob_array= []
    for i in np.arange(len(H)):
        d,h = compute_likelihood(dataset, H[i])
        posterioir = priors[i] * h
        posterior_prob_array.append(posterioir)

    posterior_prob_array_normalized = posterior_prob_array / np.sum(posterior_prob_array)
    return posterior_prob_array_normalized

Hvalue = np.arange(0,8)

dataset0 = compute_posterior(D0,H,priors)
axes[0][0].bar(Hvalue ,dataset0 )
axes[0][0].set_title('D0')
axes[0][0].set_xlabel('x-value')
axes[0][0].set_ylabel('y-value')


dataset1 = compute_posterior(D1,H,priors)
axes[0][1].bar(Hvalue ,dataset1)
axes[0][1].set_title('D1')
axes[0][1].set_xlabel('x-value')
axes[0][1].set_ylabel('y-value')


dataset2 = compute_posterior(D2,H,priors)
axes[1][0].bar(Hvalue,dataset2)
axes[1][0].set_title('D2')
axes[1][0].set_xlabel('x-value')
axes[1][0].set_ylabel('y-value')


dataset3 = compute_posterior(D3,H,priors)
axes[1][1].bar(Hvalue ,dataset3 )
axes[1][1].set_title('D3')
axes[1][1].set_xlabel('x-value')
axes[1][1].set_ylabel('y-value')

dataset4 = compute_posterior(D4,H,priors)
axes[2][0].bar(Hvalue ,dataset4)
axes[2][0].set_title('D4')
axes[2][0].set_xlabel('x-value')
axes[2][0].set_ylabel('y-value')

dataset5 = compute_posterior(D5,H,priors)
axes[2][1].bar(Hvalue ,dataset5)
axes[2][1].set_title('D5')
axes[2][1].set_xlabel('x-value')
axes[2][1].set_ylabel('y-value')


dataset6 = compute_posterior(D6,H,priors)
axes[3][0].bar(Hvalue ,dataset6)
axes[3][0].set_title('D6')
axes[3][0].set_xlabel('x-value')
axes[3][0].set_ylabel('y-value')

dataset7  = compute_posterior(D7,H,priors)
axes[3][1].bar(Hvalue ,dataset7)
axes[3][1].set_title('D7')
axes[3][1].set_xlabel('x-value')
axes[3][1].set_ylabel('y-value')


figure.suptitle("Posterior Probability of each Hypothesis")

plt.tight_layout()

figure.savefig("PS8_Q3.png")


# ## Q4 - Posterior predictive
#
# _"Yesterday's posterior is today's prior."
# $${\displaystyle p({\tilde {d}}|\mathbf {D} )= \sum_{h\in H} p({\tilde {d}}|h ,\mathbf {D} )\,p(h |\mathbf {D})}.$$
#
# `Posterior_predictive` is a function that will return the posterior predictive probability of each number from 1 to 100 marginalized over 8 hypotheses.
# In[9]:

def posterior_predictive(numbers, data, h_space, priors):
    posteriors = compute_posterior(data, h_space, priors)
    likelihood_array = []
    for hypothesis in h_space:
        likelihood_array.append(compute_likelihood(numbers, hypothesis)[0])
    newposteriors = [
        np.array(likelihood) * posterior
        for likelihood, posterior in zip(likelihood_array, posteriors)
    ]
    marginalized_value = np.array(newposteriors).sum(axis=0)

    return marginalized_value


# In[10]:

figure, axes = plt.subplots(4, 2, sharex='col', sharey='row', figsize=(16, 10))
numbers = np.arange(1,101)
Hvalue = np.arange(0,8)

axes[0][0].bar(x, posterior_predictive(numbers,D0,H,priors))
axes[0][0].set_title('D0')
axes[0][0].set_xlabel('x-value')
axes[0][0].set_ylabel('y-value')
axes[0][0].set_ylim([0,0.2])


axes[0][1].bar(x , posterior_predictive(numbers,D1,H,priors))
axes[0][1].set_title('D1')
axes[0][1].set_xlabel('x-value')
axes[0][1].set_ylabel('y-value')
axes[0][1].set_ylim([0,0.2])


axes[1][0].bar(x,posterior_predictive(numbers,D2,H,priors))
axes[1][0].set_title('D2')
axes[1][0].set_xlabel('x-value')
axes[1][0].set_ylabel('y-value')
axes[1][0].set_ylim([0,0.2])

axes[1][1].bar(x ,posterior_predictive(numbers,D3,H,priors))
axes[1][1].set_title('D3')
axes[1][1].set_xlabel('x-value')
axes[1][1].set_ylabel('y-value')
axes[1][1].set_ylim([0,0.2])


axes[2][0].bar(x, posterior_predictive(numbers,D4,H,priors))
axes[2][0].set_title('D4')
axes[2][0].set_xlabel('x-value')
axes[2][0].set_ylabel('y-value')
axes[2][0].set_ylim([0,0.2])

axes[2][1].bar(x, posterior_predictive(numbers,D5,H,priors))
axes[2][1].set_title('D5')
axes[2][1].set_xlabel('x-value')
axes[2][1].set_ylabel('y-value')
axes[2][1].set_ylim([0,0.2])

axes[3][0].bar(x ,posterior_predictive(numbers,D6,H,priors))
axes[3][0].set_title('D6')
axes[3][0].set_xlabel('x-value')
axes[3][0].set_ylabel('y-value')
axes[3][0].set_ylim([0,0.2])


axes[3][1].bar(x, posterior_predictive(numbers,D7,H,priors))
axes[3][1].set_title('D7')
axes[3][1].set_xlabel('x-value')
axes[3][1].set_ylabel('y-value')
axes[3][1].set_ylim([0,0.2])

plt.suptitle("Posterior Probability of each Hypothesis")
plt.tight_layout()

figure.savefig("PS8_Q4.png")


# ## Q5 - Testing other hypotheses
# Here we will re-make the plots from earlier but now incorporate "range-based hypotheses".
#
# To do this, let's assume that the 8 hypotheses H0-H7 each have a prior of 1/9 and the remaining 1/9th of the total probability is distributed equally among all intervals in the range 1-100. Here we will define an "interval" as something containing two distinct points, such as [50-51] or [3-88] (first number is smaller than the second), but not [31]. To calculate each number's likelihood within each interval, keep on using the size principle prior.

# In[11]:

import itertools

intervals = np.arange(1,101)
H = [H0, H1, H2, H3, H4, H5, H6, H7]
def permutation(intervals):
    list_permutations = list(itertools.permutations(intervals,2))
    permutations =[]
    for i in list_permutations:
        if i[0] < i[1]:
            permutations.append(i)
    return np.array(permutations)

interval_list = []
for i in permutation(intervals):
    interval_list.append(np.arange(i[0],i[1]+1))

H_list = H.copy()
for i in interval_list:
    H_list.append(i)

print(H_list)

updated_priors = []
for i in range(8):
    updated_priors.append(1/9)
for i in range(4950):
    updated_priors.append((1/9)/4950)


updated_priors = np.array(updated_priors)
H_list = np.array(H_list)


# In[14]:


figure, axes = plt.subplots(4, 2, sharex='col', sharey='row', figsize=(16, 10))

x = np.arange(1,101)

axes[0][0].bar(x, posterior_predictive(x, D0, new_H ,new_priors))
axes[0][0].set_title('D0')
axes[0][0].set_xlabel('x-value')
axes[0][0].set_ylabel('y-value')
axes[0][0].set_ylim([0,0.2])

axes[0][1].bar(x, posterior_predictive(x, D1,new_H ,new_priors))
axes[0][1].set_title('D1')
axes[0][1].set_xlabel('x-value')
axes[0][1].set_ylabel('y-value')
axes[0][1].set_ylim([0,0.2])

axes[1][0].bar(x,posterior_predictive(x, D2, new_H,new_priors))
axes[1][0].set_title('D2')
axes[1][0].set_xlabel('x-value')
axes[1][0].set_ylabel('y-value')
axes[1][0].set_ylim([0,0.2])


axes[1][1].bar(x, posterior_predictive(x, D3, new_H ,new_priors))
axes[1][1].set_title('D3')
axes[1][1].set_xlabel('x-value')
axes[1][1].set_ylabel('y-value')
axes[1][1].set_ylim([0,0.2])


axes[2][0].bar(x, posterior_predictive(x, D4, new_H ,new_priors))
axes[2][0].set_title('D4')
axes[2][0].set_xlabel('x-value')
axes[2][0].set_ylabel('y-value')
axes[2][0].set_ylim([0,0.2])


axes[2][1].bar(x ,posterior_predictive(x, D5, new_H ,new_priors))
axes[2][1].set_title('D5')
axes[2][1].set_xlabel('x-value')
axes[2][1].set_ylabel('y-value')
axes[2][1].set_ylim([0,0.2])

axes[3][0].bar(x ,posterior_predictive(x, D6, new_H ,new_priors))
axes[3][0].set_title('D6')
axes[3][0].set_xlabel('x-value')
axes[3][0].set_ylabel('y-value')
axes[3][0].set_ylim([0,0.2])


axes[3][1].bar(x ,posterior_predictive(x, D7, new_H ,new_priors))
axes[3][1].set_title('D7')
axes[3][1].set_xlabel('x-value')
axes[3][1].set_ylabel('y-value')
axes[3][1].set_ylim([0,0.2])

figure.suptitle("Posterior Probability of each Hypothesis")
plt.tight_layout()

# figure.savefig("PS8_Q5.png")


# <div style="background-color: #c1f2a5">
#
# </div>
#
#
# <center>
#   <img src="https://www.dropbox.com/s/7s189m4dsvu5j65/instruction.png?dl=1" width="300"/>
# </center>
#
# <div style="background-color: #c1f2a5">
#
# - Submit the `.py` file you just created in Gradescope's PS7-code.
#
# </div>
#
#
#
#
# </div>
#

# ---
