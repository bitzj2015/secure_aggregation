import torch
import torch.nn.functional as F
class Mine(torch.nn.Module):
    def __init__(self, input_size=2, hidden_size=32):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, 1)
        self.bn = torch.nn.BatchNorm1d(hidden_size)
        self.bn2 = torch.nn.BatchNorm1d(hidden_size)
        torch.nn.init.normal_(self.fc1.weight,std=0.01)
        torch.nn.init.constant_(self.fc1.bias, 0)
        torch.nn.init.normal_(self.fc2.weight,std=0.01)
        torch.nn.init.constant_(self.fc2.bias, 0)
        torch.nn.init.normal_(self.fc3.weight,std=0.01)
        torch.nn.init.constant_(self.fc3.bias, 0)
        
    def forward(self, input):
        output = F.relu(self.bn(self.fc1(input)))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output

def mutual_information(joint, marginal, mine_net):
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et

def learn_mine(batch, device, mine_net, mine_net_optim, ma_et, ma_rate=0.05):
    # batch is a tuple of (joint, marginal)
    joint , marginal = batch["joint"].to(device), batch["margin"].to(device)
    # joint = torch.autograd.Variable(torch.FloatTensor(joint)).to(device)
    # marginal = torch.autograd.Variable(torch.FloatTensor(marginal)).to(device)
    mi_lb , t, et = mutual_information(joint, marginal, mine_net)
    ma_et = (1-ma_rate)*ma_et + ma_rate*torch.mean(et)
    
    # unbiasing use moving average
    loss = -(torch.mean(t) - (1/ma_et.mean()).detach()*torch.mean(et))
    # use biased estimator
#     loss = - mi_lb
    
    mine_net_optim.zero_grad()
    torch.autograd.backward(loss)
    mine_net_optim.step()
    return mi_lb, ma_et