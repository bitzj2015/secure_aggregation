# Measure the privacy leakage in federated learning with secure aggregation
Ahmed Roushdy Elkordy, Jiang Zhang, Yahya H. Ezzeldin, Konstantinos Psounis, Salman Avestimehr. How Much Privacy Does Federated Learning with Secure Aggregation Guarantee? PETS, 2023 ([Download paper](https://arxiv.org/abs/2208.02304)).



## Usage of MINE
```
from mine import *

# Define hyperparameters
num_iter = 1000

# Define MINE network
mine_net = Mine(input_size=grad_dim * 2).to(device)
mine_net_optim = torch.optim.Adam(mine_net.parameters(), lr=0.01)
mine_net.train()


# Get joint distributino of X and Y
X = np.array(sample_individual_grad_concatenated)
Y = np.array(sample_grad_aggregate_concatenated)
joint = torch.from_numpy(np.concatenate([X, Y], axis=1).astype("float32"))

# Get marginal distribution of Y as Y_ 
random.shuffle(sample_grad_aggregate_concatenated)
Y_ = np.array(sample_grad_aggregate_concatenated)
margin = torch.from_numpy(np.concatenate([X, Y_], axis=1).astype("float32"))

# Define MINE dataset
mine_dataset = MINEDataset(joint, margin)
mine_traindataloader = DataLoader(mine_dataset, batch_size=args.mine_batch_size, shuffle=True)

# Train MINE network
for niter in range(num_iter):
    mi_lb_sum = 0
    ma_et = 1
    for i, batch in enumerate(mine_traindataloader):
        mi_lb, ma_et = learn_mine(batch, device, mine_net, mine_net_optim, ma_et)
        mi_lb_sum += mi_lb
    if niter % 10 == 0:
        logger.info(f"MINE iter: {niter}, MI estimation: {mi_lb_sum / (i+1)}")
```
