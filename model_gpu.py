import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import ray
from copy import deepcopy
torch.manual_seed(0)

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


@ray.remote
class Worker(object):
    def __init__(self, local_dataloader, lr, model_name="fcnn", dataset_name="mnist", device="cpu", algo="fedprox"):
        self.device = device
        if model_name == "fcnn":
            self.local_model = FCNNModel(dataset_name).to(device)
        elif model_name == "linear":
            self.local_model = LinearModel(dataset_name).to(device)
        elif model_name == "nlinear":
            self.local_model = NonLinearModel(dataset_name).to(device)
        else:
            self.local_model = AlexNet(dataset_name).to(device)
        
        self.grad_dim = sum(p.numel() for p in self.local_model.parameters())
        # print(self.grad_dim)
        self.local_loss = torch.nn.CrossEntropyLoss()
        self.local_optimizer = torch.optim.SGD(self.local_model.parameters(), lr=lr, momentum=0.9)
        self.global_optim_state = {'optimizer_state_dict': deepcopy(self.local_optimizer.state_dict()), 'model_state_dict': deepcopy(self.local_model.state_dict())}
        self.lr = lr
        self.local_dataloader = local_dataloader
        self.algo = algo
    
    def get_grad_dim(self):
        return int(self.grad_dim)

    def run_train_epoch(self, ep=1, update_global_state=False):
        self.local_model.train()
        self.local_optimizer.load_state_dict(self.global_optim_state['optimizer_state_dict'])
        self.local_model.load_state_dict(self.global_optim_state['model_state_dict'])
        # print(self.local_model.state_dict()["network.5.running_var"])
        for _ in range(ep):
            for _, batch in enumerate(self.local_dataloader):
                x, y = batch["x"].to(self.device), batch["y"].to(self.device)
                self.local_optimizer.zero_grad()
                outputs = self.local_model(x)
                if self.algo == "fedprox":
                    proximal_term = 0.0
                    for name, param in self.local_model.named_parameters():
                        proximal_term += (param - self.global_optim_state['model_state_dict'][name]).norm(2)
                    loss = self.local_loss(outputs, y) + (0.03/ 2) * proximal_term
                else:
                    loss = self.local_loss(outputs, y)
                loss.backward()
                self.local_optimizer.step()
                if self.algo == "fedsgd":
                    break
                # break
        if update_global_state:
            self.global_optim_state['optimizer_state_dict'] = deepcopy(self.local_optimizer.state_dict())
            # self.global_optim_state['model_state_dict'] = deepcopy(self.local_model.state_dict())
        return
    
    def pull_global_model(self, global_model_param):
        with torch.no_grad():
            for name, param in self.local_model.named_parameters():
                param.data = global_model_param[name].clone().detach()
        self.global_optim_state['model_state_dict'] = deepcopy(self.local_model.state_dict())
        return

    def push_local_model_updates(self, old_global_model_param):
        local_model_updates = {}
        with torch.no_grad():
            for name, param in self.local_model.named_parameters():
                local_model_updates[name] = param.data.clone().detach() - old_global_model_param[name].clone().detach()
        return local_model_updates

    def get_local_model_param(self):
        local_model_param = {}
        with torch.no_grad():
            for name, param in self.local_model.named_parameters():
                local_model_param[name] = param.data.clone().detach()
        return local_model_param

    def run_eval_epoch(self, testdataloader):
        self.local_model.eval()
        count = 0
        acc = 0
        for _, batch in enumerate(testdataloader):
            x, y = batch["x"].to(self.device), batch["y"].to(self.device)
            outputs = self.local_model(x)
            _, predicted_val = torch.max(outputs.data, 1)
            count += y.size(0)
            correct = (predicted_val == y).sum().item()
            acc += correct
        # print(outputs[0],y)
        return acc / count

    def random_transform(self, img):
        rand_angle = torch.randint(-180, 180, (1, )).item()
        aug_img = torchvision.transforms.functional.rotate(img, angle=rand_angle)
        return aug_img

