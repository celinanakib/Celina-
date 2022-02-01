#!/usr/bin/env python
# coding: utf-8

# <div style="background-color: #c1f2a5">
#
#
#
# Here we will work with graphical models to reason from data to hypothesis.

# </div>

# In[1]:


import numpy as np


#
#   Bayesian networks provide an efficient way of representing a probability distribution. This made them very popular in artificial intelligence research, because they make it easy to handle uncertainty, as well as inferences both from cause to effect (making predictions) and effect to cause (forming explanations). These factors traditionally presented a challenge for AI approaches based on rules and symbols, such as production systems.

#   The figure below shows a simple graphical model relating three variables. $X$ indicates whether the sprinklers were on last night, $Y$ indicates whether it rained last night, and $Z$ indicates whether the grass is wet. All three variables take on two values, with the value 1 being yes, and 0 being no.
#
#
# </div>
#
#
#
#
# <center>
#   <img src="https://www.dropbox.com/s/y67y8owgvi63ogo/sprinklers.png?raw=1" width="300"/>
# </center>
#
#
#

# In this model, $X$, and $Y$ are <i>Bernoulli</i>-distributed random variables. A Bernoulli random variable takes a value of 1 with sucess probability $\theta$ and a value of 0 with failure probability $1-\theta$, where $\theta \in [0,1]$. For example, for $X$, we could write $P(X=1)=\theta_X$ and $P(X=0)=1-\theta_X$. Similarly, for $Y$, we can write $P(Y=1)=\theta_Y$ and $P(Y=0)=1-\theta_Y$.
#
# In[2]:


def bernoulli(value, theta):
    """
    Returns the probability that this variable will take on the
    specified value given success probability theta.

    Parameters
    ----------

    value: integer
        Value of the random variable (either 0 or 1)
    theta : float
        Probability of success (between 0 and 1)

    Returns
    -------
    a float corresponding to the the probability of realizing the value

    """
    # YOUR CODE HERE
    if value == 1:
        return theta
    else:
        return 1 - theta


# In[3]:

print("P(X=0 | theta=0.2) = " + str(bernoulli(0, .2)))
print("P(X=1 | theta=0.2) = " + str(bernoulli(1, .2)))
print("P(Y=0 | theta=0.4) = " + str(bernoulli(0, .4)))
print("P(Y=1 | theta=0.4) = " + str(bernoulli(1, .4)))


# In[5]:


"""Check that `bernoulli` produces expected output."""
from nose.tools import assert_equal

for theta in np.linspace(0, 1, 100):
    assert_equal(bernoulli(0, theta), 1 - theta)
    assert_equal(bernoulli(1, theta), theta)

print("Success!")


# ---
# ## Q2 - Computing the joint probability
#
# In the current case, we will say that the probability that $X=1$ (that the sprinkler is on) is $0.6$, and the probability that $Y=1$ (that it rained) is $0.2$. In other words, $\theta_X = 0.6$ and $\theta_Y = 0.2$.
#
# Below, we define two functions `p_x` and `p_y` by calling the `bernoulli` function you just wrote, using $0.6$ and $0.2$ as the theta values respectively. Additionally, we will say that the probability of $Z$ given $X$ and $Y$ (the probability of the state of the grass, given whether it rained and whether the sprinkler is on) is given in this table (Table 1):
#
#
# <table class="table table-striped" style="width: 32em;">
#     <thead>
#     <tr>
#         <th>$x$</th>
#         <th>$y$</th>
#         <th style="white-space: nowrap; width:20em">$P(~Z=1~|~X=x,Y=y~)$</th>
#     </tr>
#     </thead>
#     <tbody>
#     <tr>
#         <td>0</td>
#         <td>0</td>
#         <td>0.05</td>
#     </tr>
#     <tr>
#         <td> 0 </td>
#         <td> 1 </td>
#         <td> 1.0 </td>
#     </tr>
#     <tr>
#         <td> 1 </td>
#         <td> 0 </td>
#         <td> 1.0 </td>
#     </tr>
#     <tr>
#         <td> 1 </td>
#         <td> 1 </td>
#         <td> 1.0 </td>
#     </tr>
#     </tbody>
# </table>
#
# The provided function `p_z_given_xy` returns probability that
# $Z=z$ for a given combination of $x$ and $y$, following the above table:

# In[6]:


def p_x(x):
    """Computes P(X=x)"""
    return bernoulli(x, 0.6)

def p_y(y):
    """Computes P(Y=y)"""
    return bernoulli(y, 0.2)

def p_z_given_xy(z, x, y):
    """Computes P(Z=z | X=x, Y=y)"""
    if x == 0 and y == 0:
        return bernoulli(z, 0.05)
    else:
        return bernoulli(z, 1)


# Given the above information about $P(X=x)$, $P(Y=y)$, and $P(Z=z\ |\ X=x,X=y)$, it is now possible to derive the joint probability distribution on $X$, $Y$, and $Z$ in order to populate the fourth column for the following table, Table 2.
#
# <table class="table table-striped" style="width: 20em;">
#     <thead>
# 	<tr>
# 		<th> $x$ </th>
# 		<th> $y$ </th>
# 		<th> $z$ </th>
# 		<th style="white-space: nowrap; width:10em"> $P(~x,y,z~)$ </th>
# 	</tr>
#     </thead>
#     <tbody>
# 	<tr>
# 		<td> 0 </td>
# 		<td> 0 </td>
# 		<td> 0 </td>
# 		<td></td>
# 	</tr>
# 	<tr>
# 		<td> 0 </td>
# 		<td> 0 </td>
# 		<td> 1 </td>
# 		<td></td>
# 	</tr>
# 	<tr>
# 		<td> 0 </td>
# 		<td> 1 </td>
# 		<td> 0 </td>
# 		<td></td>
# 	</tr>
# 	<tr>
# 		<td> 0 </td>
# 		<td> 1 </td>
# 		<td> 1 </td>
# 		<td></td>
# 	</tr>
# 	<tr>
# 		<td> 1 </td>
# 		<td> 0 </td>
# 		<td> 0 </td>
# 		<td></td>
# 	</tr>
# 	<tr>
# 		<td> 1 </td>
# 		<td> 0 </td>
# 		<td> 1 </td>
# 		<td></td>
# 	</tr>
# 	<tr>
# 		<td> 1 </td>
# 		<td> 1 </td>
# 		<td> 0 </td>
# 		<td></td>
# 	</tr>
# 	<tr>
# 		<td> 1 </td>
# 		<td> 1 </td>
# 		<td> 1 </td>
# 	<td></td>
# 	</tr>
#     </tbody>
# </table>

# ### Q2.1 Joint probability formula
# <div class="alert alert-success">
#
# $$
# P(𝑋,𝑌,𝑍)=P(𝑋 = 𝑥)P(𝑌 = 𝑦)P(𝑍 = 𝑧|𝑋 = 𝑥,𝑌 = 𝑦)
# $$
#

# ### Q2.2. Joint probability code
# <div class="alert alert-success">Now, complete the function `joint` so that it returns the joint probability as you defined it above. Use the functions `p_x`, `p_y`, and `p_z_given_xy` in the function definition, but do *not* call the function `bernoulli` (values of $\theta$ are defined in `p_x`, `p_y`, and `p_z_given_xy`). </div>
#
#
# To emphasize, `p_x`, `p_y`, and `p_z_given_xy` are *functions* that you can call (choosing and using the correct inputs, of course).
#
# Now we can compute the values for Table 2 by applying the `joint` function to each combination of $x$, $y$, and $z$ values in the next cell. Report the output of this cell in gradescope, as well as your code for the `joint` function.
#

# In[7]:


def joint(x, y, z):
    """
    Returns the joint probability distribution P(X=x, Y=y, Z=z)
    for the values x, y, and z.

    Parameters
    ----------
    x, y, z: integer
        Value of random variable X, Y, and Z, respectively

    Returns
    -------
    a float corresponding to the probability

    """
    return p_x(x)*p_y(y)*p_z_given_xy(z,x,y)


# In[8]:


# add your own test cases here!


# In[9]:


"""Check that `joint` produces expected output."""
from numpy.testing import assert_allclose

# first assume that p_x, p_y, and p_z_given_xy have the specified values
assert_allclose(joint(0, 0, 0), 0.304)
assert_allclose(joint(1, 1, 1), 0.12)

# check that the p_x, p_y, and p_z_given_xy functions are called
old_p_x = p_x
old_p_y = p_y
old_p_z_given_xy = p_z_given_xy
del p_x
del p_y
del p_z_given_xy
try:
    joint(0, 0, 1)
except NameError:
    pass
else:
    raise AssertionError("joint does not call one or more of the p_x, p_y, and/or p_z_given_xy functions")
finally:
    p_x = old_p_x
    p_y = old_p_y
    p_z_given_xy = old_p_z_given_xy
    del old_p_x
    del old_p_y
    del old_p_z_given_xy

# switch out p_x, p_y, and p_z_given_xy with alternative theta values
old_p_x = p_x
old_p_y = p_y
old_p_z_given_xy = p_z_given_xy
def p_x(x):
    return bernoulli(x, 0.5)
def p_y(y):
    return bernoulli(y, 0.3)
def p_z_given_xy(z, x, y):
    if x == 0 and y == 0:
        return bernoulli(z, 0.8)
    else:
        return bernoulli(z, 0.7)

try:
    assert_allclose(joint(0, 0, 0), 0.07)
    assert_allclose(joint(0, 0, 1), 0.28)
    assert_allclose(joint(0, 1, 0), 0.045)
    assert_allclose(joint(0, 1, 1), 0.105)
    assert_allclose(joint(1, 0, 0), 0.105)
    assert_allclose(joint(1, 0, 1), 0.245)
    assert_allclose(joint(1, 1, 0), 0.045)
    assert_allclose(joint(1, 1, 1), 0.105)
finally:
    p_x = old_p_x
    p_y = old_p_y
    p_z_given_xy = old_p_z_given_xy
    del old_p_x, old_p_y, old_p_z_given_xy

print("Success!")


# In[10]:


# Copy the output of this cell in gradescope
for x in [0, 1]:
    for y in [0, 1]:
        for z in [0, 1]:
            print("P(X={}, Y={}, Z={}) = {}".format(x, y, z, joint(x, y, z)))


# ### Q2.3. Joint probability discussion
# If you coded the `joint` function right, the cell above with test cases should return `success!`. Furthermore, three events should have a probability of $0$. Say in words which three events these are, and explain in words why that is so, based on what the random variables mean (e.g. sprinkler etc.) and what the graphical model is.

# ---
# ## Q3  Conditional probability [5 pts total]
#
# Imagine that you observed the grass is wet in the morning ($Z=1$). What happens to your beliefs about the sprinklers being on and whether it rained? To answer this question we need to compute the conditional distribution over $X$ and $Y$ when $Z=z$, $P(X=x,Y=y\ |\ Z=z)$. We can calculate the conditional distribution from the joint distribution $P(X=x, Y=y, Z=z)$ as follows:
#
# $$
# P(X=x,Y=y\ |\ Z=z)=\frac{P(X=x, Y=y, Z=z)}{\sum_{x^\prime\in\{0, 1\}}\sum_{y^\prime\in \{0, 1\}} P(X=x^\prime, Y=y^\prime, Z=z)}
# $$
#
# Note that the denominator of this equation is equivalent to $P(Z=z)$.

#
# ## Q3.1 Conditional probability function

# In[12]:


def p_xy_given_z(x, y, z):
    """
    Returns the conditional probability distribution P(X=x, Y=y | Z=z)
    for the values x, y, and z.

    Parameters
    ----------
    x, y, z: integers
        values of random variables X, Y, and Z, respectively

    Returns
    -------
    a float corresponding to the probability

    """
    # YOUR CODE HERE
    prob_z = []
    numerator = joint(x,y,z)
    for f in [0, 1]:
        for g in [0, 1]:
            prob_z.append(joint(f, g, z))

    sum_probability = sum(prob_z)
    conditional_probability = numerator/sum_probability

    return conditional_probability


# In[13]:


"""Check that `p_xy_given_z` produces expected output."""

assert_allclose(p_xy_given_z(0, 0, 0), 1.0, atol=1e-05)
assert_allclose(p_xy_given_z(0, 1, 1), 0.11494, atol=1e-05)
assert_allclose(p_xy_given_z(1, 0, 1), 0.68965, atol=1e-05)

# check that it calls joint
old_joint = joint
del joint
try:
    p_xy_given_z(0, 0, 1)
except NameError:
    pass
else:
    raise AssertionError("p_xy_given_z does not call the joint function")
finally:
    joint = old_joint
    del old_joint

# switch out p_x, p_y, and p_z_given_xy with alternative theta values
old_p_x = p_x
old_p_y = p_y
old_p_z_given_xy = p_z_given_xy
def p_x(x):
    return bernoulli(x, 0.5)
def p_y(y):
    return bernoulli(y, 0.3)
def p_z_given_xy(z, x, y):
    if x == 0 and y == 0:
        return bernoulli(z, 0.8)
    else:
        return bernoulli(z, 0.7)

try:
    assert_allclose(p_xy_given_z(0, 0, 0), 0.2641509433962263, atol=1e-05)
    assert_allclose(p_xy_given_z(0, 1, 0), 0.169811320754717, atol=1e-05)
    assert_allclose(p_xy_given_z(1, 0, 0), 0.39622641509433965, atol=1e-05)
    assert_allclose(p_xy_given_z(1, 1, 0), 0.169811320754717, atol=1e-05)
    assert_allclose(p_xy_given_z(0, 0, 1), 0.380952380952381, atol=1e-05)
    assert_allclose(p_xy_given_z(0, 1, 1), 0.14285714285714288, atol=1e-05)
    assert_allclose(p_xy_given_z(1, 0, 1), 0.33333333333333337, atol=1e-05)
    assert_allclose(p_xy_given_z(1, 1, 1), 0.14285714285714288, atol=1e-05)
finally:
    p_x = old_p_x
    p_y = old_p_y
    p_z_given_xy = old_p_z_given_xy
    del old_p_x, old_p_y, old_p_z_given_xy

print("Success!")


#

# ## Q3.2 Conditional probability discussion
# A) Once we complete `p_xy_given_z`, we can compute the probabilities for each value of $x$, $y$, and $z$:
#
# B) If  output is correct, you should see that $P(X = 0, Y=0|Z=0) = 1$.
#
# C) If your output is correct, $P(X = 0, Y=1|Z=1)1$ should be smaller than $P(X = 1, Y=0|Z=1)1$ based on the generative model

# In[14]:


for z in [0, 1]:
    for x in [0, 1]:
        for y in [0, 1]:
            print("P(X={}, Y={} | Z={}) = {}".format(x, y, z, p_xy_given_z(x, y, z)))


# ---
# ## Q4. Marginalizing
#
#
# ### Q4.1. Equation
# Now let's say that we are just interested in the belief about whether the sprinklers were on, given that we observed that the grass was wet ($Z=1$).

#
# $$
# P(X=x|Z=z)=\frac{\sum_{y'\in\{0,1\}}P(X=x,Y=y',Z=z)}{\sum_{x'\in\{0, 1\}}\sum_{y'\in \{0, 1\}} P(X=x', Y=y', Z=z)}
# $$

#
# ### Q4.2. p_x_given_z
# In[25]:

def p_x_given_z(x, z):
    """
    Returns the marginal probability distribution P(X=x | Z=z) given
    the values x and z.

    Parameters
    ----------
    x, z : integers
        values of the random variables X and Z, respectively

    Returns
    -------
    a float corresponding to the probability

    """
    # YOUR CODE HERE

    prob_z = []
    numerator = []
    for f in [0, 1]:
        numerator.append(joint(x,f,z))

    for f in [0, 1]:
        for g in [0, 1]:
            prob_z.append(joint(f, g, z))

    sum_numerator = sum(numerator)
    sum_denominator = sum(prob_z)
    probability = sum_numerator/sum_denominator

    return probability


# In[27]:


"""Check that `p_x_given_z` produces expected output."""

# check that it calls joint
old_joint = joint
del joint
try:
    p_x_given_z(0, 1)
except NameError:
    pass
else:
    raise AssertionError("p_x_given_z does not call the joint function")
finally:
    joint = old_joint
    del old_joint

# switch out p_x, p_y, and p_z_given_xy with alternative theta values
old_p_x = p_x
old_p_y = p_y
old_p_z_given_xy = p_z_given_xy
def p_x(x):
    return bernoulli(x, 0.5)
def p_y(y):
    return bernoulli(y, 0.3)
def p_z_given_xy(z, x, y):
    if x == 0 and y == 0:
        return bernoulli(z, 0.8)
    else:
        return bernoulli(z, 0.7)

try:
    assert_allclose(p_x_given_z(1, 1), 0.4761904761904762, atol=1e-05)
    assert_allclose(p_x_given_z(0, 1), 0.5238095238095238, atol=1e-05)
    assert_allclose(p_x_given_z(1, 0), 0.5660377358490567, atol=1e-05)
    assert_allclose(p_x_given_z(0, 0), 0.43396226415094336, atol=1e-05)
finally:
    p_x = old_p_x
    p_y = old_p_y
    p_z_given_xy = old_p_z_given_xy
    del old_p_x, old_p_y, old_p_z_given_xy

print("Success!")


#
# Once we implement `p_x_given_z`, we can print out the probability table for all values of $X$ and $Z$.
#

# In[28]:


for z in [0, 1]:
    for x in [0, 1]:
        print("P(X={} | Z={}) = {}".format(x, z, p_x_given_z(x, z)))


# In[29]:

print("P(X=1)       = {}".format(p_x(1)))
print("P(X=1 | Z=1) = {}".format(p_x_given_z(1, 1)))


# YOUR ANSWER HERE

# ---
# ## Q5
#
# Imagine you got into your car, and heard on the radio that it rained last night ($Y=1$). We're going to see how does this affect your beliefs about the sprinklers being on
#
#
# ### Q5.1 Equation
#
# $$
# P(X=x|Y=y, Z=z)=\frac{P(X=x, Y=y, Z=z)}{\sum_{x'\in\{0, 1\}} P(X=x', Y=y, Z=z)}
# $$


# In[30]:


def p_x_given_yz(x, y, z):
    """
    Returns the conditional probability distribution P(X=x | Y=y, Z=z)
    given the values x, y, and z.

    Parameters
    ----------
    x, y, z : integers
        values of the random variables X, Y, and Z

    Returns
    -------
    a float corresponding to the probability

    """
    # YOUR CODE HERE
    if joint(x,y,z)==0:
        return(0)
    else:
        return(joint(x,y,z)/(joint(0,y,z)+joint(1,y,z)))



# In[32]:


"""Check that `p_x_given_yz` produces expected output."""


# check that it calls joint
old_joint = joint
del joint
try:
    p_x_given_yz(0, 1, 1)
except NameError:
    pass
else:
    raise AssertionError("p_x_given_yz does not call the joint function")
finally:
    joint = old_joint
    del old_joint

# switch out p_x, p_y, and p_z_given_xy with alternative theta values
old_p_x = p_x
old_p_y = p_y
old_p_z_given_xy = p_z_given_xy
def p_x(x):
    return bernoulli(x, 0.5)
def p_y(y):
    return bernoulli(y, 0.3)
def p_z_given_xy(z, x, y):
    if x == 0 and y == 0:
        return bernoulli(z, 0.8)
    else:
        return bernoulli(z, 0.7)

try:
    assert_allclose(p_x_given_yz(0, 0, 0), 0.4, atol=1e-05)
    assert_allclose(p_x_given_yz(1, 0, 0), 0.6, atol=1e-05)
    assert_allclose(p_x_given_yz(0, 1, 0), 0.5, atol=1e-05)
    assert_allclose(p_x_given_yz(1, 1, 0), 0.5, atol=1e-05)
    assert_allclose(p_x_given_yz(0, 0, 1), 0.5333333333333333, atol=1e-05)
    assert_allclose(p_x_given_yz(1, 0, 1), 0.4666666666666667, atol=1e-05)
    assert_allclose(p_x_given_yz(0, 1, 1), 0.5, atol=1e-05)
    assert_allclose(p_x_given_yz(1, 1, 1), 0.5, atol=1e-05)
finally:
    p_x = old_p_x
    p_y = old_p_y
    p_z_given_xy = old_p_z_given_xy
    del old_p_x, old_p_y, old_p_z_given_xy

print("Success!")


# After implementing `p_x_given_yz`, we can print out the full probability table:

# In[33]:
for z in [0, 1]:
    for y in [0, 1]:
        for x in [0, 1]:
            print("P(X={} | Y={}, Z={}) = {}".format(x, y, z, p_x_given_yz(x, y, z)))


# In[34]:

print("P(X=1 | Z=1)      = {}".format(p_x_given_z(1, 1)))
print("P(X=1 | Y=1, Z=1) = {}".format(p_x_given_yz(1, 1, 1)))
print("P(X=1)            = {}".format(p_x(1)))


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
