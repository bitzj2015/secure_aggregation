import torch
import ray

class LinearModel(torch.nn.Module):

    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc1 = torch.nn.Linear(784, 10)
        torch.nn.init.uniform_(self.fc1.weight, a=0.0, b=0.01)
        torch.nn.init.uniform_(self.fc1.bias, a=0.0, b=0.01)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        return x

class FCNNModel(torch.nn.Module):
    def __init__(self):
        super(FCNNModel, self).__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 10),
        )

    def forward(self, x):
        x = x.view(x.size(0), 28 * 28)
        x = self.classifier(x)
        return x

@ray.remote
class Worker(object):
    def __init__(self, local_dataloader, lr, model_name="fcnn"):
        if model_name == "fcnn":
            self.local_model = FCNNModel()
            self.grad_dim = 89610
        elif model_name == "linear":
            self.local_model = LinearModel()
            self.grad_dim = 7850
        else:
            self.local_model = None
            self.grad_dim = 0
        self.local_loss = torch.nn.CrossEntropyLoss()
        self.local_optimizer = torch.optim.SGD(self.local_model.parameters(), lr=lr, momentum=0)
        self.lr = lr
        self.local_dataloader = local_dataloader
    
    def get_grad_dim(self):
        return int(self.grad_dim)

    def run_train_epoch(self):
        self.local_model.train()
        for k, batch in enumerate(self.local_dataloader):
            x, y = batch["x"], batch["y"]
            self.local_optimizer.zero_grad()
            outputs = self.local_model(x)
            loss = self.local_loss(outputs, y)
            loss.backward()
            self.local_optimizer.step()
        return
    
    def pull_global_model(self, global_model_param):
        with torch.no_grad():
            for name, param in self.local_model.named_parameters():
                param.data = global_model_param[name].clone().detach()
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
            x, y = batch["x"], batch["y"]
            outputs = self.local_model(x)
            _, predicted_val = torch.max(outputs.data, 1)
            count += y.size(0)
            correct = (predicted_val == y).sum().item()
            acc += correct
        return acc / count
