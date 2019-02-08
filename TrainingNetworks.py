# At first the network is naive, it doesn't know the function mapping the inputs to the outputs. 
# We train the network by showing it examples of real data, then adjusting the network parameters 
# such that it approximates this function.

# To find these parameters, we need to know how poorly the network is predicting the real outputs. 
# For this we calculate a loss function (also called the cost), a measure of our prediction error. 
# For example, the mean squared loss is often used in regression and binary classification problems

# ℓ=1/2𝑛 ∑𝑖𝑛(𝑦𝑖−𝑦̂ 𝑖)2
 
# where  𝑛  is the number of training examples,  𝑦𝑖  are the true labels, and  𝑦̂ 𝑖  are the predicted labels.

# By minimizing this loss with respect to the network parameters, we can find configurations where the loss 
# is at a minimum and the network is able to predict the correct labels with high accuracy. 
# We find this minimum using a process called gradient descent. 
# he gradient is the slope of the loss function and points in the direction of fastest change. 
# To get to the minimum in the least amount of time, we then want to follow the gradient (downwards).
# You can think of this like descending a mountain by following the steepest slope to the base.
from collections import OrderedDict

import numpy as np
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import helper
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
# Download and load the training data
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


# I'll build a network with `nn.Sequential` here. 
# Only difference from the last part is I'm not actually using softmax on the output, 
# but instead just using the raw output from the last layer. 
# This is because the output from softmax is a probability distribution. 
# Often, the output will have values really close to zero or really close to one. 
# Due to [inaccuracies with representing numbers as floating points]
# (https://docs.python.org/3/tutorial/floatingpoint.html), computations with a softmax output 
# can lose accuracy and become unstable. To get around this, 
# we'll use the raw output, called the **logits**, to calculate the loss.



# Hyperparameters for our network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                      ('relu2', nn.ReLU()),
                      ('logits', nn.Linear(hidden_sizes[1], output_size))]))


# ## Training the network!
# 
# The first thing we need to do for training is define our loss function. 
# In PyTorch, you'll usually see this as `criterion`. Here we're using softmax output, 
# so we want to use `criterion = nn.CrossEntropyLoss()` as our loss. 
# Later when training, you use `loss = criterion(output, targets)` to calculate the actual loss.
# 
# We also need to define the optimizer we're using, SGD or Adam, 
# or something along those lines. Here I'll just use SGD with `torch.optim.SGD`, 
# passing in the network parameters and the learning rate.


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# --------------------------------------------------------------------------------# 
# First, let's consider just one learning step before looping through all the data. 
# The general process with PyTorch:
#   * Make a forward pass through the network to get the logits 
#   * Use the logits to calculate the loss
#   * Perform a backward pass through the network with `loss.backward()` to calculate the gradients
#   * Take a step with the optimizer to update the weights
# 
# Below I'll go through one training step and print out the weights and gradients so you can see how it changes.

print('Initial weights - ', model.fc1.weight)

images, labels = next(iter(trainloader)) # get first batch of images and labels
images.resize_(64, 784)

# Clear the gradients, do this because gradients are accumulated
optimizer.zero_grad()

# Forward pass, then backward pass, then update weights
output = model.forward(images)
loss = criterion(output, labels)
loss.backward()
print('Gradient -', model.fc1.weight.grad)
optimizer.step()

print('Updated weights - ', model.fc1.weight)


# ### Training for real
# 
# Now we'll put this algorithm into a loop so we can go through all the images. 
# This is fairly straightforward. We'll loop through the mini-batches in our dataset, 
# pass the data through the network to calculate the losses, get the gradients, then run the optimizer.

optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 7
print_every = 40
steps = 0
for e in range(epochs):
    running_loss = 0
    for images, labels in iter(trainloader):
        steps += 1
        # Flatten MNIST images into a 784 long vector
        images.resize_(images.size()[0], 784)
        
        optimizer.zero_grad()
        
        # Forward and backward passes
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}".format(running_loss/print_every))
            
            running_loss = 0


# With the network trained, we can check out it's predictions.

images, labels = next(iter(trainloader))

img = images[0].view(1, 784)
# Turn off gradients to speed up this part
with torch.no_grad():
    logits = model.forward(img)

# Output of the network are logits, need to take softmax for probabilities
ps = F.softmax(logits, dim=1)
helper.view_classify(img.view(1, 28, 28), ps)


# Now our network is brilliant. It can accurately predict the digits in our images. Next up you'll write the code for training a neural network on a more complex dataset.

