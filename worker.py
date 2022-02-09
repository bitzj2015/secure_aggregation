from model import *
import ray
import torch
import torchvision


@ray.remote
class Worker(object):
    def __init__(self, local_dataloader, lr, model_name="fcnn", dataset_name="mnist", device="cpu"):
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
        self.lr = lr
        self.local_dataloader = local_dataloader
    
    def get_grad_dim(self):
        return int(self.grad_dim)

    def run_train_epoch(self):
        self.local_model.train()
        for k, batch in enumerate(self.local_dataloader):
            x, y = batch["x"].to(self.device), batch["y"].to(self.device)
            self.local_optimizer.zero_grad()
            outputs = self.local_model(x)
            loss = self.local_loss(outputs, y)
            loss.backward()
            self.local_optimizer.step()
            # break
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
            x, y = batch["x"].to(self.device), batch["y"].to(self.device)
            outputs = self.local_model(x)
            _, predicted_val = torch.max(outputs.data, 1)
            count += y.size(0)
            correct = (predicted_val == y).sum().item()
            acc += correct
        return acc / count

    def random_transform(self, img):
        rand_angle = torch.randint(-180, 180, (1, )).item()
        aug_img = torchvision.transforms.functional.rotate(img, angle=rand_angle)
        return aug_img

    def estimate_cov_mat(self, aug_factor=10):
        grad_sample_set = []
        self.local_model.train()
        for _, batch in enumerate(self.local_dataloader):
            x, y = batch["x"].to(self.device), batch["y"].to(self.device)
            # todo: check if we can run it in batchss
            for i in range(x.size(0)):
                for _ in range(aug_factor):
                    self.local_optimizer.zero_grad()
                    outputs = self.local_model(self.random_transform(x[i:i+1]))
                    loss = self.local_loss(outputs, y[i:i+1])
                    grad_sample = torch.autograd.grad(loss, self.local_model.parameters(), retain_graph=True)
                    grad_sample = torch.cat([t.reshape(-1) for t in grad_sample], dim=0)
                    grad_sample_set.append(grad_sample)
     
        print(f"Gradient sample has size: {grad_sample.size()}, "
                         f"total number of gradient samples: {len(grad_sample_set)}")
        
        grad_sample_set = torch.stack(grad_sample_set, dim=0)

        print(f"Gradient sample mat has size: {grad_sample_set.size()}")

        # Get covariance matrix from a set of gradient samples
        cov_mat = grad_sample_set.t().matmul(grad_sample_set) / (grad_sample_set.size(0) - 1)
        print(f"Covariance mat has size: {cov_mat.size()}")

        # Get the eigenvalues of covariance matrix
        _, s, _ = torch.svd(cov_mat)
        print(s[0:5])

        return torch.min(s).item()

