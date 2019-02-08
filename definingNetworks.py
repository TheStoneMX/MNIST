# Import things like usual

import numpy as np
import torch

import helper

import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# First up, we need to get our dataset. 
# This is provided through the torchvision package. 
# The code below will download the MNIST dataset, 
# then create training and test datasets for us. 
# Don't worry too much about the details here, you'll learn more about this later.

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])
# Download and load the training data
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# We have the training data loaded into trainloader and we make that an iterator with iter(trainloader). 
# We'd use this to loop through the dataset for training, 
# but here I'm just grabbing the first batch so we can check out the data. 
# We can see below that images is just a tensor with size (64, 1, 28, 28). 
# So, 64 images per batch, 1 color channel, and 28x28 images.

dataiter = iter(trainloader)
images, labels = dataiter.next()

# show random image.
plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')
plt.show()

# Here I'll use PyTorch to build a simple feedfoward network to classify the MNIST images. 
# That is, the network will receive a digit image as input and predict the digit in the image

# To build a neural network with PyTorch, you use the torch.nn module. 
# The network itself is a class inheriting from torch.nn.Module. 
# You define each of the operations separately, 
# like nn.Linear(784, 128) for a fully connected linear layer with 784 inputs and 128 units.

# The class needs to include a forward method that implements the forward pass through the network. 
# In this method, you pass some input tensor x through each of the operations you defined earlier. 
# The torch.nn module also has functional equivalents for things like ReLUs in torch.nn.functional. 
# This module is usually imported as F. 
# Then to use a ReLU activation on some layer (which is just a tensor), you'd do F.relu(x). 
# Below are a few different commonly used activation functions.
# Sigmoid - TanH - ReLu


from torch import nn
from torch import optim
import torch.nn.functional as F

###################################################
# this is the long way to build a network class
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Defining the layers, 128, 64, 10 units each
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 128)
        self.fc3 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.fc4 = nn.Linear(64, 10)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.softmax(x, dim=1)
         
        return x

model = Network()
model
# print out ouf the network class
# Network(
#   (fc1): Linear(in_features=784, out_features=500, bias=True)
#   (fc2): Linear(in_features=500, out_features=128, bias=True)
#   (fc3): Linear(in_features=128, out_features=64, bias=True)
#   (fc4): Linear(in_features=64, out_features=10, bias=True)
# )
#####################################################################

###------------------------------------------------------------###
# This is the Pytorch way to build network class
# PyTorch provides a convenient way to build networks like this where a tensor is passed sequentially through operations, 
# nn.Sequential. Using this to build the equivalent network:

# Hyperparameters for our network
input_size = 784
hidden_sizes = [500,128, 64]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                      nn.ReLU(),                      
                      nn.Linear(hidden_sizes[2], output_size),
                      nn.Softmax(dim=1))
print(model)

# Forward pass through the network and display output
images, labels = next(iter(trainloader))
images.resize_(images.shape[0], 1, 784)
ps = model.forward(images[0,:])
helper.view_classify(images[0].view(1, 28, 28), ps)

# Print out of nn.Sequencial
# Sequential(
#   (0): Linear(in_features=784, out_features=500, bias=True)
#   (1): ReLU()
#   (2): Linear(in_features=500, out_features=128, bias=True)
#   (3): ReLU()
#   (4): Linear(in_features=128, out_features=64, bias=True)
#   (5): ReLU()
#   (6): Linear(in_features=64, out_features=10, bias=True)
#   (7): Softmax()
# )

###------------------------------------------------------------###

###------------------------------------------------------------###
# You can also pass in an OrderedDict to name the individual layers and operations. 
# Note that a dictionary keys must be unique, so each operation must have a different name.
# Hyperparameters for our network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

from collections import OrderedDict
model = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                      ('relu2', nn.ReLU()),
                      ('output', nn.Linear(hidden_sizes[1], output_size)),
                      ('softmax', nn.Softmax(dim=1))]))
model

# Forward pass through the network and display output
images, labels = next(iter(trainloader))
images.resize_(images.shape[0], 1, 784)
ps = model.forward(images[0,:])
helper.view_classify(images[0].view(1, 28, 28), ps)
###------------------------------------------------------------###


# Initializing weights and biases
# The weights and such are automatically initialized for you, 
# but it's possible to customize how they are initialized. 
# The weights and biases are tensors attached to the layer you defined, 
# you can get them with model.fc1.weight for instance.

print(model.fc1.weight)
print(model.fc1.bias)

# For custom initialization, we want to modify these tensors in place. 
# These are actually autograd Variables, so we need to get back the actual tensors with model.fc1.weight.data. 
# Once we have the tensors, we can fill them with zeros (for biases) or random normal values.

# Set biases to all zeros
model.fc1.bias.data.fill_(0)

# sample from random normal with standard dev = 0.01
model.fc1.weight.data.normal_(std=0.01)

# Forward pass
# Now that we have a network, let's see what happens when we pass in an image. 
# This is called the forward pass. We're going to convert the image data into a tensor, 
# then pass it through the operations defined by the network architecture.

# Grab some data 
dataiter = iter(trainloader) # I'm just grabbing the first batch so we can check out the data. 
images, labels = dataiter.next()
# or 
images, labels = next(iter(trainloader))

# AFTER we get out batch of images we need to resize them
# Resize images into a 1D vector, new shape is (batch size, color channels, image pixels) 
# images.resize_(64, 1, 784)
# or images.resize_(images.shape[0], 1, 784) to not automatically get batch size
images.resize_(images.shape[0], 1, 784)

# Forward pass through the network
img_idx = 0
ps = model.forward(images[img_idx,:])

# print(images.shape) ### torch.Size([64, 1, 28, 28])
# print(labels.shape) ### torch.Size([64])

img = images[img_idx]
helper.view_classify(img.view(1, 28, 28), ps)