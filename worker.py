from model import *
import ray
import torch
import torchvision
from copy import deepcopy
import constant
print(constant.cpu_per_worker, constant.gpu_per_worker)

@ray.remote(num_cpus=constant.cpu_per_worker, num_gpus=constant.gpu_per_worker)
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
                    loss = self.local_loss(outputs, y) + (0.01/ 2) * proximal_term
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
