#!/usr/bin/env python
# coding: utf-8

# ---
# <div style="background-color: #c1f2a5">
#
#
#Here, we are going to learn to navigate a simple maze.
#
# </div>

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# We are going to learn to navigate a simple maze. In this maze, you have to make two successive left-right decisions (0 or 1). The first takes you from state $s=0$ to states $s=1$ or $s=2$. The second decision takes you to states $3-6$, as pictured below.
# ![](Maze_JDS.png)
# Taking a first step to S1 or S2 is costly ($r=-1$), but can get you to a position to get reward (states 3 and 6 give you 4, and 10 points respectively). The goal is to gain as many points as possible, so you should learn to select right-right, which will give you a total cumulated reward of $-1+10 = 9$.
#
# We will try a few different algorithms to learn this, and see how they behave differently.
#
# The reward function and the transition function are below
# - R[i,j] indicates what reward you obtain for selecting action j in state i
# - T[i,j] indicates what state you will be in after you select action j in state i

# In[2]:

R = np.array([[-1,-1],[4,-1],[-1,10]])
T = np.array([[1,2],[3,4],[5,6]])

# ## Q1 - Softmax choice
#
# Softmax equation:
# $$ P(a|s) = \frac{exp(\beta Q(s,a)}{\sum_i exp(\beta Q(s,a_i))}$$
#
# $$ P(a|s) = \frac{1}{\sum_i exp(\beta [Q(s,a_i) - Q(s,a)])}$$
#
# In[3]:

def softmax(beta,Qs):
    """
    Returns softmax probability of a choice.

    Parameters
    ----------
    beta : real number
        The softmax inverse temperature parameter
    Qs: a (1,n) numpy array of values

    Returns
    -------
    a: an integer in [0,n]
        a choice made with probability defined by softmax(betaQs)
    """
    Qs_array = np.empty(len(Qs))
    for i in range(len(Qs)):
        Qs_array[i]= 1/(np.sum(np.exp(beta*(Qs-Qs[i]))))
    return int(np.random.choice(np.arange(len(Qs)), p = Qs_array))

# In[4]:

# In[5]:


"""Checksoftmax computes the correct values"""
from numpy.testing import assert_allclose

assert_allclose(softmax(100,np.array([0,1])), 1.0)
assert_allclose(softmax(100,np.array([0,1,0])), 1.0)
assert_allclose(softmax(100,np.array([0,0,1])), 2.0)
assert_allclose(softmax(100,np.array([1,0,0,0])), 0.0)

print("Success!")


# In[6]:

def left_right_bias(beta,Qs,n_trials):
    """
    Returns average cumulated reward for a left-right bias agent.

    Parameters
    ----------
    beta : real number
        The softmax inverse temperature parameter
    Qs: a (1,n) numpy array of values
    n_trials: integer - number of trials to average over

    Returns
    -------
    float
        average cumulated reward
    """
    Cum_R = np.zeros(n_trials)
    for i in range(n_trials):
        # start at state 0
        initial_state = 0
        # make the first choice
        first_choice = softmax(beta,Qs)
        # see what reward that state gives, and what the next state is
        first_reward = R[initial_state,first_choice]
        next_state = T[initial_state,first_choice]
        # make a second choice
        second_choice = softmax(beta,Qs)
        # get reward
        second_reward =  R[next_state,second_choice]
        # store cumulated reward in this trial
        Cum_R[i] =first_reward + second_reward
    # return average cumulated reward.
    return np.mean(Cum_R)

n_trials = 10000
beta = 0
Qs = np.zeros(2)
print('Random choice leads to average cumulated reward of: '+str(np.around(left_right_bias(beta,Qs,n_trials),decimals=2)))
beta = 5
Qs = np.array([.8,.5])
print('Random left-biased choice leads to average cumulated reward of: '+str(np.around(left_right_bias(beta,Qs,n_trials),decimals=2)))
beta = 5
Qs = np.array([.5,.8])
print('Random right-biased choice leads to average cumulated reward of: '+str(np.around(left_right_bias(beta,Qs,n_trials),decimals=2)))
beta = 50
Qs = np.array([.5,.8])
print('Near-greedy right-biased choice leads to average cumulated reward of: '+str(np.around(left_right_bias(beta,Qs,n_trials),decimals=2)))


# ## Q2. SARSA
#
# Now, we're going to code an agent that actually learns the values of different choices.
# We'll use the SARSA equation:
# $$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha (r_t + \gamma Q(s_{t+1},a_{t+1}) - Q(s_t,a_t))$$
#
# If there's no next state/action (i.e. when the algorithm reaches the end of the maze in any of states 3-6), the equation is simply:
# $$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha (r_t  - Q(s_t,a_t))$$
#
#
# ### Q2.1 SARSA Update
# In[7]:


def sarsa(alpha, gamma,Q,s,a,r):
    """
    Returns updated Q-table.

    Parameters
    ----------
    alpha : real number
        The learning rate parameter
    gamma : real number
        The discount parameter
    Q: a (3,2) numpy array of values for states (0,1,2) and actions (0,1)

    Returns
    -------
    Q: a (3,2) numpy array of updated values for states (0,1,2) and actions (0,1)
    """

    delta = r[0] + (gamma * Q[s[1], a[1]]) - Q[s[0], a[0]]
    Q[s[0], a[0]] = Q[s[0], a[0]] + alpha * delta

    delta = r[1] + (gamma * 0) - Q[s[1], a[1]]
    Q[s[1], a[1]] = Q[s[1], a[1]] + alpha * delta

    return Q


# In[8]:

# In[9]:


"""Check sarsa computes the correct values"""
from numpy.testing import assert_allclose

s = np.array([0,1]).astype(int)
a = np.zeros(2).astype(int)
r = np.array([-1,4]).astype(int)

alpha = .1
gamma=.9
assert_allclose(sarsa(alpha, gamma,0.5*np.ones([3,2]),s,a,r), np.array([[0.395,0.5],[0.85,0.5],[0.5,0.5]]))
a = np.ones(2).astype(int)
assert_allclose(sarsa(alpha, gamma,0.5*np.ones([3,2]),s,a,r), np.array([[0.5,0.395],[0.5,0.85],[0.5,0.5]]))
s = np.array([0,2]).astype(int)
assert_allclose(sarsa(alpha, gamma,0.5*np.ones([3,2]),s,a,r), np.array([[0.5,0.395],[0.5,0.5],[0.5,0.85]]))
alpha = .2
assert_allclose(sarsa(alpha, gamma,0.5*np.ones([3,2]),s,a,r), np.array([[0.5,0.29],[0.5,0.5],[0.5,1.2]]))
Q= sarsa(alpha, gamma,0.5*np.ones([3,2]),s,a,r)
gamma=.75
assert_allclose(sarsa(alpha, gamma,Q,s,a,r), np.array([[0.5,0.212],[0.5,0.5],[0.5,1.76]]))

print("Success!")


# ### Q2.2 One trial SARSA
# Now, use the `sarsa` function you wrote as well as the `softmax` function you wrote to complete the function `onetrial_sarsa` below. This function should navigate one path through the maze, and return the two choices and rewards experienced, as well as the updated Q-value. Hint: you can use the `left_right_bias` function as a model for walking through the maze.

# In[10]:


def onetrial_sarsa(parameters,Q,R,T):
    """
    Returns updated Q-table.

    Input
    ----------
    parameters : (1,3) numpy array
        model parameters (beta, alpha, gamma)
    Q: a (3,2) numpy array of values for states (0,1,2) and actions (0,1)
    R: reward function (3,2) numpy array
    T: transition function (3,2) numpay array

    Returns
    -------
    Q: (3,2) numpy array of updated Q-values
    a: a (1,2) numpy array of the sequence of two choices
    r: a (1,2) numpy array of the sequence of two rewards.
    """
    a = []
    r = []

    initial_state = 0

    first_choice = softmax(parameters[0],Q[0])
    first_reward = R[initial_state,first_choice]

    a.append(first_choice)
    r.append(first_reward)


    next_state = T[initial_state,first_choice]

        # make a second choice

    second_choice = softmax(parameters[0],Q[1])
    second_reward = R[next_state,second_choice]

    a.append(second_choice)
    r.append(second_reward)

    Q = (sarsa(parameters[1], parameters[2],Q,s,a,r))

    return Q,a,r

# In[12]:


# Plotting the results

R = np.array([[-1,-1],[4,-1],[-1,10]])
T = np.array([[1,2],[3,4],[5,6]])


nTrials = 100
Qs = np.empty((6,nTrials))
Q = np.array([[.5,.5],[.5,.5],[.5,.5]])
beta = 5
alpha = .1
gamma = 1
parameters = np.array([beta,alpha,gamma])

for t in range(nTrials):
    newQ,a,r = onetrial_sarsa(parameters,Q,R,T)
    Qs[:,t] = np.ndarray.flatten(newQ)

figure, axis = plt.subplots()
for i in range(6):
    axis.plot(Qs[i,:])

axis.legend(['Q(S0,A0)','Q(S0,A1)','Q(S1,A0)','Q(S1,A1)','Q(S2,A0)','Q(S2,A1)'])
plt.title('One trial SARSA')
plt.xlabel('Trials')
plt.ylabel('Value')

figure.savefig('PS12_Q2.png')


# ## Q3. Performance
#
# Now, we'll run multiple simulations (100), and see what the final Q-values are. Use the provided parameter values (50 trials, beta=5, alpha=0.1, gamma=1). Make sure to re-initialize your Q-values for each simulation!
#
#
# ### Q3.1 Plot final Q-values
# In three subplots, plot scatterplots of a) Q(s0,a0) vs. Q(s0,a1) b) Q(s0,a1) vs. Q(s1,a0) and c) Q(s0,a1) vs. Q(s2,a1).
#
#

# In[13]:


R = np.array([[-1,-1],[4,-1],[-1,10]])
T = np.array([[1,2],[3,4],[5,6]])


nTrials = 50
niterations = 100
Qs = np.empty((6,niterations))
beta = 5
alpha = .1
gamma = 1
parameters = np.array([beta,alpha,gamma])

Q = np.array([[.5,.5],[.5,.5],[.5,.5]])

figure, axis = plt.subplots(1,3)

for t in range(niterations):
    newQ,a,r = onetrial_sarsa(parameters,Q,R,T)
    Qs[:,t] = np.ndarray.flatten(newQ)


array = []

for i in range(6):
    array.append((Qs[i,:]))

axis[0].scatter(array[0])
axis[0].plot(array[1])

for t in range(niterations):
    newQ,a,r = onetrial_sarsa(parameters,Q,R,T)
    Qs[:,t] = np.ndarray.flatten(newQ)

axis[1].plot(array[1])
axis[1].plot(array[2])


for t in range(niterations):
    newQ,a,r = onetrial_sarsa(parameters,Q,R,T)
    Qs[:,t] = np.ndarray.flatten(newQ)

axis[2].plot(array[1])
axis[2].plot(array[5])


figure.savefig('PS12_Q3.png')

# In[63]:


R = np.array([[-1,-1],[4,-1],[-1,10]])
T = np.array([[1,2],[3,4],[5,6]])


nTrials = 50
niterations = 100
Qs = np.empty((6,niterations))
beta = 5
alpha = .1
gamma = 1
parameters = np.array([beta,alpha,gamma])

Q = np.array([[.5,.5],[.5,.5],[.5,.5]])

figure, axis = plt.subplots(3)

Q = np.array([[.5,.5],[.5,.5],[.5,.5]])
for n in range(niterations):
    for t in range(nTrials):
        newQ,a,r = onetrial_sarsa(parameters,Q,R,T)
        Qs[:,t] = np.ndarray.flatten(newQ)

array0 = []
for i in range(6):
    array0.append((Qs[i,:]))

axis[0].scatter(array0[0], array0[1])

Q = np.array([[.5,.5],[.5,.5],[.5,.5]])
for n in range(niterations):
    for t in range(nTrials):
        newQ,a,r = onetrial_sarsa(parameters,Q,R,T)
        Qs[:,t] = np.ndarray.flatten(newQ)

array1 = []
for i in range(6):
    array1.append((Qs[i,:]))

axis[1].scatter(array1[1], array1[2])

Q = np.array([[.5,.5],[.5,.5],[.5,.5]])
for n in range(niterations):
    for t in range(nTrials):
        newQ,a,r = onetrial_sarsa(parameters,Q,R,T)
        Qs[:,t] = np.ndarray.flatten(newQ)

array2 = []
for i in range(6):
    array2.append((Qs[i,:]))

axis[2].scatter(array2[1], array2[5])

axis[0].set_title('Q(s0,a0) vs. Q(s0,a1)')
axis[0].set_xlabel('Q(s0,a0)')
axis[0].set_ylabel('Q(s0,a1)')



axis[1].set_title('Q(s0,a1) vs. Q(s1,a0)')
axis[1].set_xlabel('Q(s0,a1) ')
axis[1].set_ylabel('Q(s1,a0)')


axis[2].set_title('Q(s0,a1) vs. Q(s2,a1)')
axis[2].set_xlabel('Q(s0,a1)')
axis[2].set_ylabel('Q(s2,a1)')


plt.tight_layout()


# In[55]:


R = np.array([[-1,-1],[4,-1],[-1,10]])
T = np.array([[1,2],[3,4],[5,6]])


nTrials = 50
niterations = 100
Qs = np.empty((6,niterations))
beta = 5
alpha = .1
gamma = 1
parameters = np.array([beta,alpha,gamma])

Q = np.array([[.5,.5],[.5,.5],[.5,.5]])


figure, axis = plt.subplots(3)

Q = np.array([[.5,.5],[.5,.5],[.5,.5]])
for n in range(niterations):
    for t in range(nTrials):
        newQ,a,r = onetrial_sarsa(parameters,Q,R,T)
        Qs[:,t] = np.ndarray.flatten(newQ)

array0 = []
for i in range(6):
    array0.append((Qs[i,:]))

axis[0].scatter(array0[0], array0[1])
axis[0].set_title('Q(s0,a0) vs. Q(s0,a1)')
axis[0].set_xlabel('Q(s0,a0)')
axis[0].set_ylabel('Q(s0,a1)')


axis[1].scatter(array0[1], array0[2])
axis[1].set_title('Q(s0,a1) vs. Q(s1,a0)')
axis[1].set_xlabel('Q(s0,a1) ')
axis[1].set_ylabel('Q(s1,a0)')


axis[2].scatter(array0[1], array0[5])
axis[2].set_title('Q(s0,a1) vs. Q(s2,a1)')
axis[2].set_xlabel('Q(s0,a1')
axis[2].set_ylabel('Q(s2,a1)')


plt.tight_layout()

# ## Q4. Parameter effects
#

# ### Q4.1 Effect of beta
#
# to investigate the effect of beta on performance, use $\gamma = 0.5$. In one figure, plot performance as a function of beta value ranging 1-20 in increments of 2, for $\alpha = 0.1$ and $\alpha=0.3$.

# In[ ]:

nTrials=20
niterations=1000

gamma=0.5


betaline=np.arange(1,20,2)
figure, axis = plt.subplots()

alpha=.1
R_beta = np.empty(len(betaline))


fig.savefig('PS12_Q4_2.png')


# ### Q4.2 Effect of alpha

nTrials=20
niterations=1000

gamma=0.5
alphaline=np.arange(.05,1,.05)


fig.savefig('PS12_Q4_4.png')


"""Check sarsa computes the correct values"""
from numpy.testing import assert_allclose

assert_allclose(onetrial_planning(np.array([1,1]),np.array([[-1,-1],[4,-1],[-1,10]]),T)[0], np.array([3,-2,-2,9]))
assert_allclose(onetrial_planning(np.array([1,.5]),np.array([[-1,-1],[4,-1],[-1,10]]),T)[0], np.array([1,-1.5,-1.5,4]))

print("Success!")


fig.savefig('PS12_Q5_1.png')



import timeit

fig.savefig('PS12_Q5_2.png')


# ---
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
