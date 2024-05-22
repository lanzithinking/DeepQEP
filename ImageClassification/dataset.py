"Dataset and neural network to fit it"

import os
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

# prepare to download
os.makedirs('./data', exist_ok=True)

def dataset(dataset_name='mnist', seed=2024):
    
    # Setting manual seed for reproducibility
    torch.manual_seed(seed)
    
    if dataset_name=='mnist':
        # load data
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])
        train_dataset = datasets.MNIST('./data/MNIST', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data/MNIST', train=False, download=True, transform=transform)
        batch_size = 1024
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        num_classes = 10
        
        # choose network
        feature_extractor = Net_mnist()
    
    elif dataset_name=='cifar10':
        # load data
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        aug_trans = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        common_trans = [transforms.ToTensor(), normalize]
        train_compose = transforms.Compose(aug_trans + common_trans)
        test_compose = transforms.Compose(common_trans)
        train_dataset = datasets.CIFAR10('./data/CIFAR10', train=True, download=True, transform=train_compose)
        test_dataset = datasets.CIFAR10('./data/CIFAR10', train=False, download=True, transform=test_compose)
        batch_size = 1024
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        num_classes = 10
        
        # choose network
        feature_extractor = Net_cifar10()
    
    return train_loader, test_loader, feature_extractor, num_classes

# define the NN for mnist
class Net_mnist(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv1 = nn.Conv2d(1, 4, 4, 1)
        # self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(9216, 128)
        self.fc1 = nn.Linear(576, 10)
        # self.fc2 = nn.Linear(128, 10)
        self.output_dims = self.fc1.out_features

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        return x#output

# define the NN for cifar10
class Net_cifar10(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.output_dims = self.fc3.out_features

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x