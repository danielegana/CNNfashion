#%%

#Now I'll try to do the same with a CNN
import torch
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from numpy import random
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from helpers import plot
import torch.nn.init as init
import re


#%%
with open("inputs.txt", 'r') as file:
        for line in file:
            matchbatch = re.search(r'batchsize\s*=\s*(\d+)', line)
            matchepoch = re.search(r'epochs\s*=\s*(\d+)', line)
            matchseed = re.search(r'seed\s*=\s*(\d+)', line)
            if matchbatch:
                batch_size=int(matchbatch.group(1))
            if matchepoch:
                epochs=int(matchepoch.group(1))
            if matchseed:
                seed=int(matchseed.group(1))
file.close()

#%%

#%%

transforms = v2.Compose([ToTensor(),
  #  v2.RandomRotation(degrees=40),
    v2.RandAugment(),
])
notransforms=v2.Compose([ToTensor(),
])

#%%

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=notransforms,
)

training_dataaug = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms,
)
# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
#%%

totaltrain=ConcatDataset((training_data,training_dataaug))


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
#%%
# Define model as a class. We're creating a new class mynet.
# mynet inherits features from a base class nn.Module that allows it to perform GPU acceleration and others
class mynet(nn.Module):
# __init__ is the "standard constructor method" of the object NeuralNetwork. 
# It defines and initializes objects of the class. 
    
    def __init__(self):
        super(mynet, self).__init__()
        # 1 input image channel (black & white), 4 output channels, 5x5 square convolution
        # kernel
        filter1=50
        filter2=100
        filter3=200
        kernel1=int(5)
        kernel2=int(3)
        kernel3=int(3)
        padding1=(kernel1-1)/2
        padding2=(kernel2-1)/2
        padding3=(kernel3-1)/2
        finalpixel=int(((28-(kernel1-1)+2*padding1)/2-(kernel2-1)+2*padding2)/2-(kernel3-1)+2*padding3)
        self.conv1 = nn.Conv2d(1, filter1, kernel1, padding=int(padding1))
        self.conv2 = nn.Conv2d(filter1, filter2, kernel2,padding=int(padding2))
        self.conv3 = nn.Conv2d(filter2, filter3, kernel3,padding=int(padding3))
        self.fc1 = nn.Linear(filter3*finalpixel*finalpixel,100)  # 5*5 from image dimension
        self.fc2 = nn.Linear(100, 10)
        self.dropout = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm2d(filter1)
        self.bn2 = nn.BatchNorm2d(filter2)
        self.bn3 = nn.BatchNorm2d(filter3)
        self.bnf1 = nn.BatchNorm1d(100)
        self.bnf2 = nn.BatchNorm1d(10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), (2, 2))
        x=self.dropout(x)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), (2, 2))
        x = F.relu(self.bn3(self.conv3(x)))
       # x=self.dropout(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.bnf1(self.fc1(x)))
      #  x= self.dropout(x)
        x = F.relu(self.bnf2(self.fc2(x)))
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    

#Defines the function to train the model
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    #Initializes the training mode of the model
    model.train()
    #enumerate creates a tuple index,data. so batch gets the index number
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        # The loss is computed against the known class label. y is an integer, pred is a 10-dimensional vector
        # with the 10 classes. 
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        #optimizer.zero_grad() zeroes out the gradient after one pass. this is to 
        #avoid accumulating gradients, which is the standard behavior
        optimizer.zero_grad()

        # Print loss every 100 batches
        if batch % 100 == 0:
            loss, current = loss.item(), batch*batch_size
            print(f"loss: {loss:>7f}  [{current:>5d}]")

# %%
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct


numseeds=0
accuracyvector=np.zeros(numseeds+1)
flag=0
# %% 
for x in range(numseeds+1):
    print("Seed run =",x)
    torch.manual_seed(seed+x)
    np.random.seed(seed+x)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = mynet().to(device)

    
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size) 
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        accuracy=100*test(test_dataloader, model, loss_fn)
        #A little code to control the learning rate
        threshold1=92
        threshold2=93
        factor=0.1
        if accuracy > threshold1 and flag==0:
        # Reduce the learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] *= factor
                print(f"Learning rate reduced to: {param_group['lr']}")
                flag=1
        if accuracy > threshold2 and flag==1:
        # Reduce the learning rate
            for param_group in optimizer.param_groups:
               # param_group['lr'] *= factor
               # print(f"Learning rate reduced to: {param_group['lr']}")
                flag=2
        print("Done!")
    accuracyvector[x]=accuracy



