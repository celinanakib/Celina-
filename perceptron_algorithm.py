#!/usr/bin/env python
# coding: utf-8

# <div style="background-color: #c1f2a5">
#
#
#
# In this problem set, we are going to implement the perceptron algorithm on a very simple feature space: pixels in images of handwritten digits.
#
# Make sure:
# - that all plots are scaled in such a way that you can see what is going on (while still respecting any specific plotting instructions).
# - that the general patterns are fairly represented.
# - to label all x- and y-axes unless otherwise instructed, and to include a title.
#
# </div>

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt


# ## Import images for classification
#
# Here we will train a perceptron to classify images of handwritten numbers 0-9.
#
# - `x.flatten()` (take a N-dimensional numpy array and make it one-dimensional)
# - `np.random.choice` -- choose from the list of images
# - `np.dot()` -- compute the dot product of two vectors: the sum of the product of the coordinates.
# - `np.reshape()` -- reshapes a vector to a matrix
# - `x.imshow()` -- visualize a matrix as an image

# In[2]:


DIM = (28,28) #these are the dimensions of the image

def load_image_files(n, path="images/"):
    # helper file to help load the images
    # returns a list of numpy vectors
    images = []
    for f in sorted(os.listdir(os.path.join(path,str(n)))): # read files in the path
        p = os.path.join(path,str(n),f)
        if os.path.isfile(p):
            i = np.loadtxt(p)
            assert i.shape == DIM # just check the dimensions here
            # i is loaded as a matrix, but we are going to flatten it into a single vector
            images.append(i.flatten())
    return images


# Load up these image files
A = load_image_files(1)
B = load_image_files(2)

N = len(A[0]) # the total size
assert N == DIM[0]*DIM[1] # just check our sizes to be sure


# set up some random initial weights
weights = np.random.normal(0,1,size=N)


# In[3]:


# Load up these image files
A = load_image_files(0)
B = load_image_files(1)

N = len(A[0]) # the total size of input features
assert N == DIM[0]*DIM[1] # just check our sizes to be sure


# ## Q1. Visualize the images [SOLO, 5 pts]
#
# Let's explore what we have loaded so far by examining the type and length of A (images of the handwritten number '0') and B (images of the handwritten number '1').

# In[4]:


print('The data type of A is:',type(A))
print('\nThe number of images in A is: ', len(A))
print('\nEach image in A is of the form: ', type(A[0]), ' with length: ', len(A[0]))
print('\nThe number of images in B is: ',len(B))


# Here we will isualize three randomly chosen examples of the "1" images, and three randomly chosen examples of the "0" images in a figure with subplots in 2 rows and 3 columns. Make sure to indicate in each subplot title which number is being shown.
#
# You will need to use the np.reshape function to put each image back into a square file. Check one of the previous problem sets on how to visualize an array as an image!
#
# Remove x and y tickmarks for every subplot. Since these are images, do not label your x- and y-axes. You should include titles on each subplot indicating which numbers are shown, and one overall figure title.
#
# Save the total figure (all subplots in one figure) image as a png (refer to previous assignments, or right click and select 'Save As' in Jupyter)... and put the png in Gradescope.
#

# In[5]:


figure, axis = plt.subplots(2,3)

#YOUR CODE HERE

choice1 = np.random.choice(np.arange(5924))
image_1 = B[choice1].reshape(28,28)

choice2 = np.random.choice(np.arange(5924))
image_2 = B[choice2].reshape(28,28)

choice3 = np.random.choice(np.arange(5924))
image_3 = B[choice3].reshape(28,28)

choice4 = np.random.choice(np.arange(5924))
image_4 = A[choice4].reshape(28,28)

choice5 = np.random.choice(np.arange(5924))
image_5 = A[choice5].reshape(28,28)

choice6 = np.random.choice(np.arange(5924))
image_6 = A[choice6].reshape(28,28)

axis[0, 0].imshow(image_1)
axis[0, 1].imshow(image_2)
axis[0, 2].imshow(image_3)
axis[1, 0].imshow(image_4)
axis[1, 1].imshow(image_5)
axis[1, 2].imshow(image_6)

axis[0, 0].get_yaxis().set_visible(False)
axis[0, 1].get_yaxis().set_visible(False)
axis[0, 2].get_yaxis().set_visible(False)
axis[1, 0].get_yaxis().set_visible(False)
axis[1, 1].get_yaxis().set_visible(False)
axis[1, 2].get_yaxis().set_visible(False)

axis[0, 0].get_xaxis().set_visible(False)
axis[0, 1].get_xaxis().set_visible(False)
axis[0, 2].get_xaxis().set_visible(False)
axis[1, 0].get_xaxis().set_visible(False)
axis[1, 1].get_xaxis().set_visible(False)
axis[1, 2].get_xaxis().set_visible(False)

axis[0, 0].set_title('Image of 1')
axis[0, 1].set_title('Image of 1')
axis[0, 2].set_title('Image of 1')
axis[1, 0].set_title('Image of 0')
axis[1, 1].set_title('Image of 0')
axis[1, 2].set_title('Image of 0')

figure.suptitle('Image Visualization')
figure.savefig('PS6_Q1.png')


# ## Q2.1 Perceptron output
#
# compute_output function that takes 1) the weights W of the perceptron and 2) an image (as a vector of 784 features), and computes the perceptron's output.
#
#

# In[6]:


def compute_output(W,image):
    """
    Computes the output of the current network with weight matrix W for one image.

    Parameters
    ----------
    W: the weight matrix (1,n)
    image:  a length n array of input features representing one image

    Returns
    -------
    the 1 or 0 output of the network

    """
    # YOUR CODE HERE

    output = (np.dot(image,W)>0).astype(int)
    return (output)



# In[7]:

A = load_image_files(0)
B = load_image_files(1)
weights = (np.arange(len(A[0]))/(len(A[0])))-.5


Q2_test = np.empty(10)
for i in range(5):
    Q2_test[2*i] = compute_output(A[i],weights)
    Q2_test[2*i + 1] = compute_output(B[i],weights)
print(Q2_test)

# In[8]:


Q2_answers = np.empty(10)
weights = np.sin(np.arange(len(A[0])))
for i in range(5):
    Q2_answers[2*i] = compute_output(A[i],weights)
    Q2_answers[2*i + 1] = compute_output(B[i],weights)

print('Q2 answer')
print(Q2_answers)


# ## Overall accuracy
# Function that computes the current accuracy of the network on a set of images.
#

# In[9]:


def compute_accuracy(W,images,labels):
    """
    Computes the network accuracy on a list of images

    Parameters
    ----------
    W: the current weight matrix
    images:  list of length n array images for number
    labels: list of correct labels for images

    Returns
    -------
    Proportion of correct labels

    """
    accuracy = 0
    for i in range(len(images)):
        output = compute_output(W, images[i])
        if output == labels[i]:
            accuracy += 1

    return (accuracy/len(images))


# In[10]:

weights = (np.arange(len(A[0]))/(len(A[0])))-.5
accuracyA = compute_accuracy(weights,A,np.zeros(len(A)))
print('accuracy on "0" images is ',accuracyA)
accuracyB = compute_accuracy(weights,B,np.zeros(len(B)))
print('accuracy on "1" images is ',accuracyB)

#accuracy on "0" images is  0.0595981765996961
#accuracy on "1" images is  0.10872144764164936


# In[11]:


# Q3 output
print('Q3_answer')

sin_weights = np.sin(np.arange(len(A[0])))
accuracyA = compute_accuracy(sin_weights,A,np.zeros(len(A)))
print('accuracy on "0" images is ',accuracyA)
accuracyB = compute_accuracy(sin_weights,B,np.zeros(len(B)))
print('accuracy on "1" images is ',accuracyB)


# ## Q4. Updating the weights after one single training example
#
# Function update_weights_single_image that updates the network's weights with one example: an image and its label (1 or 0) over one update cycle.
#
#
# Simplest perceptron algorithm:
# 1. Use the input W to compute $y = \sum_i w_i \times x_i$ and determine predicted label (1/0).
#
# 2. If predicted and true label disagree:
#     --> If true label is “1”: W += x
#     --> If true label is "0": W -= x
#    If predicted and true label agree, do nothing.
#

# In[12]:


def update_weights_single_image(W,image,label):
    """
    Updates the weight matrix W after one training (image,label) pair.


    Parameters
    ----------
    W: the current weight matrix
    image:  a length n array of input features representing one image
    label: a single number indicating whether the image is a 0 or a 1

    Returns
    -------
    the new perceptron weight matrix (same size as W)

    """
    # YOUR CODE HERE
    local_W = W
    output = compute_output(W,image)
    if output != label and label == 1:
        local_W += image
    elif output != label and label == 0:
        local_W -= image
    return local_W


# **Next:**
#
# Start with 0 weights, and train the perceptron on a single image, the first image of a "1". Visualize the trained weights the same way you visualized the images in Q1: reshaping them (28,28). Use the same label and format guidelines as for Q1.
#
# Your image should have a clear pattern. Upload it to Gradescope, and explain the pattern you see.
#

# In[13]:


weights = np.zeros(len(B[0]))
# YOUR CODE HERE
update_weights_single_image(weights,B[0],1)

figure, axis = plt.subplots()
im = axis.imshow(np.reshape(weights, [28,28]))

# set the title
axis.set_title('Weights after one 1 example');
figure.savefig('PS6_Q4.png')


# ## Q5. Train with multiple images
#
# Function update_weights_multiple_images that takes starting network weights, the images and their labels, and applies the previous training to all images in order.
#
# This function should use your previous function that updated weights based on a single image. Copy your function into gradescope.

# In[25]:


def update_weights_multiple_images(W,images,labels):
    """
    Updates the weight matrix W with multiple training (image,label) pairs.

    Parameters
    ----------
    W: the current weight matrix
    images:  list of images (each image a length n array)
    labels: list of labels (1 or 0)

    Returns
    -------
    the new perceptron weight matrix (same size as W)

    """

    # YOUR CODE HERE
    for i in range(len(images)):
        W = (update_weights_single_image(W,images[i],labels[i]))
    return W


# ## Q6. Train your perceptron!
#
# We'll train the preceptron in small batches. We first initialize the weights from a random normal distribution. We'll select N_samples images of "0"s and N_samples images of "1"s randomly, then train the weights on this small batch. We'll then iterate this for as many steps as needed.
#

# In[26]:


def train_perceptron(train_0, train_1,N_samples = 5, steps = 200):
    performance = np.empty(steps)
    train_labels = np.ones(2*N_samples)
    train_labels[0:N_samples] = np.zeros(N_samples)

    full_sample = train_0+train_1
    full_labels = np.ones(len(full_sample))
    full_labels[0:len(train_0)]=np.zeros(len(train_0))

    # set up some random initial weights
    weights = np.random.normal(0,1,size=N)
    for i in range(steps):
        examples_0 = [train_0[j] for j in np.random.choice(np.arange(len(train_0)),size=N_samples,replace = False).tolist()]
        examples_1 = [train_1[j] for j in np.random.choice(np.arange(len(train_1)),size=N_samples,replace = False).tolist()]
        examples = examples_0+examples_1

        weights = update_weights_multiple_images(weights,examples,train_labels)
        performance[i] = compute_accuracy(weights,full_sample,full_labels)

    return performance, weights


# ### Train your perceptron:
#
# Plot the performance (i.e., accuracy) as a function of time (i.e., step) for N_samples = 1, 5, and 25. Use 200 steps for each.
#

# In[27]:


figure, axis = plt.subplots(nrows=3, ncols=1, figsize=(8,12), sharex=True)

sample_1, weight_1 = train_perceptron(A,B,N_samples = 1, steps = 200)
sample_5, weight_5 = train_perceptron(A,B,N_samples = 5, steps = 200)
sample_25, weight_25 = train_perceptron(A,B,N_samples = 25, steps = 200)

axis[0].plot(sample_1)
axis[0].set_title('Performace of 1 Sample Vs. Time')
axis[0].set_ylabel("Performance (Accuracy) ")
axis[0].set_ylim([0.7, 1.2])


axis[1].plot(sample_5)
axis[1].set_title('Performace of 5 Samples Vs. Time')
axis[1].set_ylabel("Performance (Accuracy) ")
axis[1].set_ylim([0.7, 1.2])


axis[2].plot(sample_25)
axis[2].set_title('Performace of 25 Samples Vs. Time')
axis[2].set_ylabel("Performance (Accuracy) ")
axis[2].set_ylim([0.7, 1.2])

plt.xlabel("Time (steps)")
figure.suptitle('Performance as a Function of Time')

#YOUR CODE HERE
figure.savefig('PS6_Q6.png')


# ## Q7. trained weights

figure, axis = plt.subplots()

sample_25, weight_25 = train_perceptron(A,B,N_samples = 25, steps = 200)
reshaped_weights = np.reshape(weight_25, (28, 28))
plt.imshow(reshaped_weights)
plt.title("Image of Trained Weight Vector")
plt.colorbar()

figure.savefig('PS6_Q7.png')


# ## Q8. Zeroing weights

trained_weights = weight_25.copy()
T = np.argsort(np.abs(weights))
N_samples=1000

choices_A = [A[j] for j in np.random.choice(np.arange(len(A)),size = N_samples,replace = False).tolist()]
choices_B = [B[j] for j in np.random.choice(np.arange(len(B)),size = N_samples,replace = False).tolist()]

iterations = np.arange(0,len(T)+1,7)
accuracy_list = np.empty(len(iterations))
A_accuracy = np.empty(len(iterations))
B_accuracy = np.empty(len(iterations))

for i in range(len(iterations)):
    thresholdweights= trained_weights
    thresholdweights[T[:(iterations[i]+1)]] = 0

    A_accuracy[i] = compute_accuracy(thresholdweights,choices_A,np.zeros(len(choices_A)))
    B_accuracy[i] = compute_accuracy(thresholdweights,choices_B,np.ones(len(choices_B)))

    if B_accuracy[i] == 0:
        accuracy_list[i] = 0

    else:
        accuracy_list[i] = A_accuracy[i] / B_accuracy[i]


# In[24]:

figure, axis = plt.subplots()
axis.plot(iterations, A_accuracy,label = 'Accuracy Image for 0')
axis.plot(iterations, B_accuracy,label = 'Accuracy Image for 1')
axis.set_ylim(0.5,1.1)
axis.legend()

plt.xlabel("Number of weight values replaced by 0")
plt.ylabel("Accuracy ")
plt.title('Correlation b/t number of weight values closest to 0 replaced by 0 and Perceptron Accuracy')


figure.savefig('PS6_Q8.png')


# ## Q9. Classifying multiple digits
#
# Train a perceptron for each possible pair of digits using the default parameters (not just "0" vs. "1", but also "0" vs. "2", ..., "8" vs."9".
#

image_0 = load_image_files(0)
image_1 = load_image_files(1)
image_2 = load_image_files(2)
image_3 = load_image_files(3)
image_4 = load_image_files(4)
image_5 = load_image_files(5)
image_6 = load_image_files(6)
image_7 = load_image_files(7)
image_8 = load_image_files(8)
image_9 = load_image_files(9)

# Make a list with each corresponding number
number_list = [image_0,image_1,image_2,image_3,image_4,image_5,image_6,image_7,image_8, image_9]


# In[44]:


#YOUR CODE HERE

digits = 10
accuracy_matrix = np.ones([digits,digits])

for i in range(digits-1):
    im1 = number_list[i]
    for j in range(i+1,digits):
        im2 = number_list[j]
        performance, weights = train_perceptron(im1, im2)
        accuracy_matrix[i,j] = performance[-1]
        accuracy_matrix[j,i] = accuracy_matrix[i,j]

figure, axis = plt.subplots()
plt.imshow(accuracy_matrix)
axis.set_ylabel("Performance (Accuracy) ")
axis.set_xlabel("Digits")
plt.title("Digit Shape Accuracy")
plt.colorbar()

figure.savefig('PS6_Q9.png')


# <div style="background-color: #c1f2a5">
#
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
# - Submit the `.py` file you just created in Gradescope's PS6-code.
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
