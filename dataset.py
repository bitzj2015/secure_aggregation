from pickle import TRUE
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np


class TaskDataset(Dataset):
    def __init__(self, input, label, client_id=-1):
        self.input = input
        self.label = label
        self.client_id = client_id

    def __len__(self):
        return np.shape(self.label)[0]

    def __getitem__(self, idx):
        if type(idx) == torch.Tensor:
            idx = idx.item()
        x = self.input[idx]
        y = self.label[idx]
        sample = {'x': x, 'y': y}
        return sample


class MINEDataset(Dataset):
    def __init__(self, joint, margin):
        self.joint = joint
        self.margin = margin

    def __len__(self):
        return np.shape(self.joint)[0]

    def __getitem__(self, idx):
        if type(idx) == torch.Tensor:
            idx = idx.item()
        joint = self.joint[idx]
        margin = self.margin[idx]
        sample = {'joint': joint, 'margin': margin}
        return sample


def get_dataset(dataset_name, batch_size, nClients, logger):
    if dataset_name == "mnist":
        train_data = datasets.MNIST(
            root = 'data',
            train = True,                         
            transform = ToTensor(), 
            download = True,            
        )
        test_data = datasets.MNIST(
            root = 'data', 
            train = False, 
            transform = ToTensor()
        )
        xtrain = train_data.data / 255.0
        xtrain = (xtrain - 0.5) / 0.5
        ytrain = train_data.targets
        trainDataSize = xtrain.shape[0]
        logger.info(f"Load dataset: {dataset_name}, training data size: {trainDataSize}")


        xtest = test_data.data / 255.0
        xtest = (xtest - 0.5) / 0.5
        ytest = test_data.targets
        logger.info(f"Load dataset: {dataset_name}, testing data size: {xtest.shape[0]}")

        trainDataSizeFracClients = 1 / nClients
        trainDataSizeClients = np.int32(trainDataSizeFracClients * trainDataSize)

        stIndex = 0
        dataloaderByClient = []
        for iClient in range(nClients):
            
            train_dataset = TaskDataset(
                xtrain[stIndex: stIndex + trainDataSizeClients], 
                ytrain[stIndex: stIndex + trainDataSizeClients],
                client_id = iClient
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            dataloaderByClient.append(train_loader)
            stIndex = (stIndex + trainDataSizeClients)
        
        test_dataset = TaskDataset(xtest, ytest)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return dataloaderByClient, test_loader
            

