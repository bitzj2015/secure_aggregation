import torch
import random
import numpy as np
from tqdm import tqdm
import json
import argparse
from mine import *
from model import *
from worker import *
from dataset import *
import ray
import logging
import subprocess
from copy import deepcopy
import h5py
import math

subprocess.run(["mkdir", "-p", "logs"])
subprocess.run(["mkdir", "-p", "param"])
subprocess.run(["mkdir", "-p", "results"])

# 807 958 1146 1087
# 1120 1212 1043 1089
def main(args):
    # Define logger

    logging.basicConfig(
        filename=f"./logs/log_{args.version}.txt",
        filemode='w',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )
    logger=logging.getLogger() 
    logger.setLevel(logging.INFO) 

    # Define hyperparameters
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    nClients = args.total_nodes
    batch_size = args.batch_size
    trainTotalRounds = args.trainTotalRounds
    nEpochs = args.nEpochs
    subset = args.subset
    num_jobs = 1
    if subset > nClients:
        num_jobs = subset // nClients
        subset = nClients


    num_iter = 1000
    num_samples = args.num_sample
    
    # dataloaderByClient, _, nClients = get_femnist(batch_size, logger)
    # np.random.seed(1)

    mine_results = {}

    if args.resample:
        for k in range(args.k):
            dataloaderByClient, testdataloader = get_dataset(args.dataset, batch_size, nClients, logger, sampling=args.sampling, alpha=args.alpha)
            xdata = dataloaderByClient[0].dataset.input.cpu().numpy().reshape(-1, 32*32*3).astype(np.float32)
            print(xdata.shape)
            hf = h5py.File(f"./dataset/grad_data_{subset}_{k}_{args.dataset}_{args.alpha}.hdf5", "w")
            hf.create_dataset('xdata', data=xdata)
            hf.close()
    else:
        mine_results = {}
        res = []
        for k in range(args.k):
            mine_results[k] = []
            hf = h5py.File(f"./dataset/grad_data_{subset}_{k}_{args.dataset}_{args.alpha}.hdf5", "r")
            X = np.array(hf["xdata"][:])
            X = X.reshape(-1, 32*32*3)
            Y_all = list(X)
            print(X.shape, len(Y_all[0]))

            # Train MINE network
            Y = np.array([np.array(grad).reshape(-1) for grad in Y_all])

            joint = torch.from_numpy(np.concatenate([X, Y], axis=1).astype("float32"))
            random.shuffle(Y_all)
            Y_ = np.array([np.array(grad).reshape(-1) for grad in Y_all])

            margin = torch.from_numpy(np.concatenate([X, Y_], axis=1).astype("float32"))

            mine_dataset = MINEDataset(joint, margin)
            mine_traindataloader = DataLoader(mine_dataset, batch_size=args.mine_batch_size, shuffle=True)
            mine_net = Mine(input_size=X.shape[1] + Y.shape[1]).to(device)
            mine_net_optim = torch.optim.Adam(mine_net.parameters(), lr=0.0003)
            mine_net.train()

            for niter in range(num_iter):
                mi_lb_sum = 0
                ma_et = 1
                for i, batch in enumerate(mine_traindataloader):
                    mi_lb, ma_et = learn_mine(batch, device, mine_net, mine_net_optim, ma_et)
                    mi_lb_sum += mi_lb
                if niter % 10 == 0:
                    logger.info(f"MINE iter: {niter}, MI estimation: {mi_lb_sum / (i+1)}")
                if not math.isnan(mi_lb.item()) and not mi_lb.item() == float('inf'):
                    mine_results[k].append(mi_lb.item())
            res.append(max(mine_results[k]))
        print(np.mean(res))

        
parser = argparse.ArgumentParser()
parser.add_argument("--total-nodes", dest="total_nodes", type=int, default=50)
parser.add_argument("--subset", type=int, default=50)
parser.add_argument("--batch-size", dest="batch_size", type=int, default=32)
parser.add_argument("--mine-batch-size", dest="mine_batch_size", type=int, default=100)
parser.add_argument("--num-sample", dest="num_sample", type=int, default=100)
parser.add_argument("--trainTotalRounds", type=int, default=30)
parser.add_argument("--nEpochs", type=int, default=1)
parser.add_argument("--version", type=str, default="test")
parser.add_argument("--model", type=str, default="linear")
parser.add_argument("--dataset", type=str, default="femnist")
parser.add_argument("--lr", type=float, default=0.03)
parser.add_argument("--k", type=int, default=5)
parser.add_argument("--alpha", type=float, default=1)
parser.add_argument("--sampling", type=str, default="noniid")
parser.add_argument('--resample', dest="resample", default=False, action='store_true')
args = parser.parse_args()

main(args=args)