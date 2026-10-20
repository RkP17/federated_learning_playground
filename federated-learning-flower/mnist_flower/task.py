"""federated-learning-flower: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import torchvision.datasets as datasets
from torchvision.transforms import Compose, Normalize, ToTensor


# Defines the model
class MLP(nn.Module):
    """Model (simple MLP)"""

    def __init__(self):
        super(MLP, self).__init__()
        self.fcl1 = nn.Linear(784,128)
        self.fcl2 = nn.Linear(128, 64)
        self.fcl3 = nn.Linear(64,10)

    def forward(self, x):
        x = torch.flatten(x,1)
        x = F.relu(self.fcl1(x))
        x = F.relu(self.fcl2(x))
        return self.fcl3(x)

def getSizes(datasetSize, num_partitions):
    base_size = datasetSize // num_partitions
    
    # give the remaineder of the subsets to the first few clients
    remainder = datasetSize % num_partitions
    
    sizes = [
        base_size + (1 if i < remainder else 0)
        for i in range(num_partitions)
    ]
    
    return sizes

train_ds = None  # Cache Dataset for training
test_ds = None # Cache Dataset for testing

# where the clients get their data
def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition MNIST data."""
    global train_ds, test_ds
    
    if train_ds is None:
        train_ds = datasets.MNIST(
            root="../data",
            train=True,
            download=False,
            transform=ToTensor()
        )
    
    if test_ds is None:
        test_ds = datasets.MNIST(
            root="../data",
            train=False,
            download=False,
            transform=ToTensor()
        )
    
    # define generator 
    gen = torch.Generator().manual_seed(42)

    # ------------ Train split -------------- #
    # random split to partition the data between clients
    partition_train = random_split(train_ds, getSizes(len(train_ds), num_partitions), generator=gen)
    
    client_train_subset =  partition_train[partition_id]
    
    # Divide data on each node: 80% train, 20% test
    train_size = int(0.8* len(client_train_subset))
    val_size = len(client_train_subset) - train_size
    
    client_train, client_val = random_split(client_train_subset, [train_size,val_size], generator=gen) # returns a dict 
    
    # ---------------- Test split ----------------- #
    partition_test = random_split(test_ds, getSizes(len(test_ds), num_partitions), generator=gen)
    
    client_test_subset = partition_test[partition_id]
    
    
    # ------------ Construct dataloaders ---------- #
    trainloader = DataLoader(
        client_train, batch_size=batch_size, shuffle=True
    )
    
    valloader = DataLoader(
        client_val, batch_size=batch_size, shuffle=True
    )

    testloader = DataLoader(client_test_subset, batch_size=batch_size, shuffle=True)
    return trainloader, valloader, testloader


# Used to test the global model - might remove this ... plan to test the model on each client
def load_centralized_dataset():
    """Load test set and return dataloader."""
    global test_ds

    if test_ds is None:
        test_ds = datasets.MNIST(
            root="../data",
            train=False,
            download=False,
            transform=ToTensor()
        )
    # return the whole test datset
    return DataLoader(test_ds, batch_size=128)


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    
    net.to(device)  # move model to GPU if available
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    
    running_loss = 0.0
    total = 0
    
    for epoch in range(epochs):
        print(f"Starting training for epoch {epoch} ...")
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad() # clear prev gradients
            
            outputs = net(inputs)
            
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            
            total += labels.size(0)
            running_loss += loss.item() * labels.size(0)
            
            
            print(f'Loss: {running_loss / len(trainloader):.4f}')

    avg_trainloss = running_loss / total
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    
    net.to(device)
    net.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    correct, total, running_loss = 0, 0, 0.0
    
    all_predicted, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = net(inputs)
            
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            # cross entropy loss returns mean loss for a batch
            loss = loss_function(outputs, labels)
            
            running_loss += loss.item() * labels.size(0)
            
            correct += (predicted == labels).sum().item()
            
            all_predicted.extend(predicted)
            all_labels.extend(labels)
            
    accuracy = correct / total
    average_loss = running_loss / total
    return average_loss, accuracy, all_labels, all_predicted
