

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # # Defining the layers, 400, 300, 100 units each 
        # self.fc1 = nn.Linear(784, 400) 
        # self.fc2 = nn.Linear(400, 300) 
        # self.fc3 = nn.Linear(300, 100) 
        # # Output layer, 10 units - one for each digit 
        # self.fc4 = nn.Linear(100, 10)

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # x = self.fc1(x) 
        # x = F.relu(x) 
        # x = self.fc2(x) 
        # x = F.relu(x) 
        # x = self.fc3(x) 
        # x = F.relu(x) 
        # x = self.fc4(x) 
        # x = F.softmax(x, dim=1)
        # return x
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)