import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import ray

class LinearModel(torch.nn.Module):

    def __init__(self, dataset_name="mnist"):
        super(LinearModel, self).__init__()
        self.dataset_name = dataset_name
        if self.dataset_name == "mnist":
            self.fc1 = torch.nn.Linear(784, 10)
            # self.grad_dim = 7850
        elif self.dataset_name == "cifar10":
            self.fc1 = torch.nn.Linear(32*32*3, 10)
            # self.grad_dim = 30730
        else:
            print("No such dataset!")
        torch.nn.init.uniform_(self.fc1.weight, a=0.0, b=0.01)
        torch.nn.init.uniform_(self.fc1.bias, a=0.0, b=0.01)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class NonLinearModel(torch.nn.Module):
    def __init__(self, dataset_name="mnist"):
        super(NonLinearModel, self).__init__()
        self.dataset_name = dataset_name
        if self.dataset_name == "mnist":
            self.fc1 = torch.nn.Linear(784, 10)
            # self.grad_dim = 7850
        elif self.dataset_name == "cifar10":
            self.fc1 = torch.nn.Linear(32*32*3, 10)
            # self.grad_dim = 30730
        else:
            print("No such dataset!")
        torch.nn.init.uniform_(self.fc1.weight, a=0.0, b=0.01)
        # torch.nn.init.constant_(self.fc1.bias, 10.0)
        torch.nn.init.uniform_(self.fc1.bias, a=0.0, b=0.01)
        self.sigmoid = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.sigmoid(self.fc1(x))
        return x

class FCNNModel(torch.nn.Module):
    def __init__(self, dataset_name="mnist"):
        super(FCNNModel, self).__init__()
        self.dataset_name = dataset_name
        if self.dataset_name == "mnist":
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(28*28, 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, 10),
            )
            self.grad_dim = 89610
        elif self.dataset_name == "cifar10":
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(32*32*3, 100),
                torch.nn.ReLU(),
                # torch.nn.Linear(100, 100),
                # torch.nn.ReLU(),
                torch.nn.Linear(100, 10),
            )
            self.grad_dim = 318410
        else:
            print("No such dataset!")


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class AlexNet(nn.Module):
    # def __init__(self):
    #     super().__init__()
    #     self.conv1 = nn.Conv2d(3, 32, 5)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(32, 32, 5)
    #     self.fc1 = nn.Linear(32 * 5 * 5, 100)
    #     # self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(100, 10)

    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = torch.flatten(x, 1) # flatten all dimensions except batch
    #     x = F.relu(self.fc1(x))
    #     # x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x

    def __init__(self, dataset_name="mnist"):
        super().__init__()
        self.dataset_name = dataset_name
        if self.dataset_name == "mnist":
            self.in_ch = 1
            self.out_dim = 64 * 3 * 3
        else:
            self.in_ch = 3
            self.out_dim = 64 * 4 * 4

        self.network = nn.Sequential(
            nn.Conv2d(self.in_ch, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 16 x 16 x 16
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 32 x 8 x 8
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 4 x 4
            nn.BatchNorm2d(64),
            nn.Flatten())
        self.fc = nn.Linear(self.out_dim, 10)
        
    def forward(self, xb):
        xb = self.network(xb)
        return self.fc(xb)