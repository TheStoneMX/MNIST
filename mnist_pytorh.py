

import torch
import torchvision
from torchvision import transforms

from models import Net
import torch.nn.functional as F
import torch.optim as optim


n_epochs = 10
batch_size_train = 64
batch_size_test = 1000
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


train_loader = torch.utils.data.DataLoader(

  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               transforms.RandomHorizontalFlip(), # horizontally flip image with probability=0.5
                               transforms.ToTensor(),   # convert the PIL Image to a tensor
                               # doing this normalization we do not get better performance. 
#                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])), 
#                             # Test set: Avg. loss: 0.1151, Accuracy: 9697/10000 (96%)                          
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))])), # Best - Test set: Avg. loss: 0.0887, Accuracy: 9725/10000 (97%) lr=0.001 10 epochs
                               batch_size=batch_size_train, shuffle=True)
                             

  # torchvision.datasets.MNIST('/files/', train=True, download=True,
  #                            transform=torchvision.transforms.Compose([
  #                              transforms.RandomHorizontalFlip(), # horizontally flip image with probability=0.5
  #                              torchvision.transforms.ToTensor(),   # convert the PIL Image to a tensor
  #                              torchvision.transforms.Normalize((0.1307,), (0.3081,))])),   
  #                              batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
  transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
  batch_size=batch_size_test, shuffle=True)


examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

print(example_data.shape)

import matplotlib.pyplot as plt

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
fig.show()

''' here we are using a SGD optimizer
Then oin another file we will use Adam
'''

learning_rate = 0.001
momentum = 0.5
use_cuda = torch.cuda.is_available()

# use Cuda-
device = torch.device("cuda" if use_cuda else "cpu")

nn = Net()
model = Net().to(device)

# optimizer = optim.SGD(nn.parameters(), lr=learning_rate, momentum=momentum) # Test set: Avg. loss: 0.2529, Accuracy: 9224/10000 (92%)

#------------- optimizer with Adam -----------------------------#
# Test set: Avg. loss: 0.2290, Accuracy: 9358/10000 (93%) lr=0.01  3 epochs
# Test set: Avg. loss: 0.1363, Accuracy: 9567/10000 (95%) lr=0.001 3 epochs
# Test set: Avg. loss: 0.0887, Accuracy: 9725/10000 (97%) lr=0.001 10 epochs
# Test set: Avg. loss: 0.1294, Accuracy: 9623/10000 (96%) lr=0.005 10 epochs
optimizer = torch.optim.Adam(nn.parameters(), lr=learning_rate) 

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

#--------------------------------#
def train(epoch):
  nn.train()

  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = nn(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()

    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))

      train_losses.append(loss.item())

      train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

      #torch.save(nn.state_dict(), 'model.pth') # '/results/model.pth'
      #torch.save(optimizer.state_dict(), 'optimizer.pth') # '/results/optimizer.pth'

#--------------------------------#
def test():
  nn.eval()
  test_loss = 0
  correct = 0

  with torch.no_grad():
    for data, target in test_loader:
      output = nn(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()

  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)

  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format( test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

#--------------------------------#
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()
  
    
