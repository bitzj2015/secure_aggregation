from pickle import TRUE
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
from skimage.measure.entropy import shannon_entropy


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
        x = x.reshape(x.size(0), x.size(1), -1).permute(2, 0, 1)
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


def get_dataset(dataset_name, batch_size, nClients, logger, sampling="iid", alpha=100):
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
        logger.info(f"Load dataset: {dataset_name}, testing data size: {xtest.shape}")

    elif dataset_name == "cifar10":
        train_data = datasets.CIFAR10(
            root = 'data',
            train = True,                         
            transform = ToTensor(), 
            download = True,            
        )
        test_data = datasets.CIFAR10(
            root = 'data', 
            train = False, 
            transform = ToTensor()
        )

        xtrain = torch.from_numpy(train_data.data.astype("float32")) / 255.0
        xtrain = (xtrain - 0.5) / 0.5
        ytrain = torch.from_numpy(np.array(train_data.targets))
        trainDataSize = xtrain.shape[0]
        logger.info(f"Load dataset: {dataset_name}, training data size: {trainDataSize}")

        xtest = torch.from_numpy(test_data.data.astype("float32")) / 255.0
        xtest = (xtest - 0.5) / 0.5
        ytest = torch.from_numpy(np.array(test_data.targets))
        logger.info(f"Load dataset: {dataset_name}, testing data size: {xtest.shape}")

    trainDataSizeFracClients = 1 / nClients
    trainDataSizeClients = np.int32(trainDataSizeFracClients * trainDataSize)
    print(sampling)
    if sampling == "iid":
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
    
    elif sampling == "gniid":
        dataloaderByClient = []
        dataset_by_class = {}
        num_data_per_user = trainDataSize // nClients

        for i in range(len(xtrain)):
            label = ytrain[i].item()
            if label not in dataset_by_class.keys():
                dataset_by_class[label] = []
            dataset_by_class[label].append(xtrain[i].numpy())

        for iClient in range(nClients):
            dist = list(np.random.rand(10))
            # dist = [np.exp(value) for value in dist]
            dist = [int(value / sum(dist) * num_data_per_user) for value in dist]
            train_x = []
            train_y = []
            for label in range(len(dist)):
                index = np.random.choice([value for value in range(len(dataset_by_class[label]))], dist[label])
                for id in index:
                    train_x.append(dataset_by_class[label][id])
                    train_y.append(label)

            train_dataset = TaskDataset(
                torch.from_numpy(np.array(train_x)), 
                torch.from_numpy(np.array(train_y)),
                client_id = iClient
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            dataloaderByClient.append(train_loader)
    else:
        dataloaderByClient = []
        dataset_by_class = {}
        num_data_per_user = trainDataSize // nClients

        for i in range(len(xtrain)):
            label = ytrain[i].item()
            if label not in dataset_by_class.keys():
                dataset_by_class[label] = []
            dataset_by_class[label].append(xtrain[i].numpy())

        proportions = []
        for label in dataset_by_class.keys():
            # dirichlet
            proportions.append(np.random.dirichlet(np.repeat(alpha, nClients)))
        for iClient in range(nClients):
            train_x = []
            train_y = []
            for label in range(len(proportions)):
                proportion = proportions[label]
                label_list_len = len(dataset_by_class[label])
                start_index = int(sum(proportion[: iClient]) * label_list_len)
                num_index = int(proportion[iClient] * label_list_len)
                for id in range(num_index):
                    train_x.append(dataset_by_class[label][id + start_index])
                    train_y.append(label)

            train_dataset = TaskDataset(
                torch.from_numpy(np.array(train_x)), 
                torch.from_numpy(np.array(train_y)),
                client_id = iClient
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            dataloaderByClient.append(train_loader)
    test_dataset = TaskDataset(xtest, ytest)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return dataloaderByClient, test_loader



# train_data = datasets.EMNIST(
#     root = 'data',
#     train = True,                         
#     transform = ToTensor(), 
#     download = True,
#     split ="byclass"          
# )
# test_data = datasets.EMNIST(
#     root = 'data', 
#     train = False, 
#     transform = ToTensor(),
#     split ="byclass"
# )
# print(train_data)

# target_user_entropy = 0
# for img in train_data.data[:1200]:
#     for k in range(img.shape[-1]):
#         target_user_entropy += shannon_entropy(img[:,:,k], base=2)
# print(target_user_entropy, 1200)