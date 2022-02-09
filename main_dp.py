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

subprocess.run(["mkdir", "-p", "logs"])
subprocess.run(["mkdir", "-p", "param"])
subprocess.run(["mkdir", "-p", "results"])

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

    np.random.seed(0)
    clientSubSet = np.random.choice(nClients-1, subset-1, replace=True)
    clientSubSet = [0] + [item + 1 for item in clientSubSet]

    dataloaderByClient, testdataloader = get_dataset(args.dataset, batch_size, nClients, logger)
    
    ray.init()
    workerByClient = [
        Worker.remote(
            dataloaderByClient[clientSubSet[i]], 
            lr=args.lr, 
            model_name=args.model, 
            dataset_name=args.dataset,
            device=device
        ) 
        for i in range(len(clientSubSet))
    ]

    global_model_param = ray.get(workerByClient[0].get_local_model_param.remote())
    grad_dim = ray.get(workerByClient[0].get_grad_dim.remote())
    mine_results = {}

    for iRound in range(trainTotalRounds):
        mine_results[iRound] = {}
        ray.get([worker.pull_global_model.remote(global_model_param) for worker in workerByClient])
        acc = ray.get(workerByClient[0].run_eval_epoch.remote(testdataloader))
        logger.info(f"Round: {iRound}, test accuracy: {acc}")

        local_model_updates = ray.get([worker.push_local_model_updates.remote(global_model_param) for worker in workerByClient])
        for name in global_model_param.keys():
            cur_param = torch.cat([item[name].unsqueeze(0) for item in local_model_updates], axis=0)
            global_model_param[name] += torch.mean(cur_param, axis=0)
        
        # Start iteration of MINE
        sample_individual_grad_concatenated = []
        sample_grad_aggregate_concatenated = []
        local_model_updates = []
        for _ in range(num_jobs):
            # Get the global model
            ray.get([worker.pull_global_model.remote(global_model_param) for worker in workerByClient])

            # Get covariance matrix of gradients
            res = ray.get([worker.estimate_cov_mat.remote(aug_factor=50) for worker in workerByClient])
            print(res)
            # Run local training epochs
            for _ in range(nEpochs):
                ray.get([worker.run_train_epoch.remote() for worker in workerByClient])

            # Get local model updates
            tmp_local_model_updates = ray.get([worker.push_local_model_updates.remote(global_model_param) for worker in workerByClient])
            local_model_updates += deepcopy(tmp_local_model_updates)
        # assert(len(local_model_updates) == nClients * num_jobs)
        individual_grad_concatenated = []
        grad_aggregate_concatenated = []

        for name in global_model_param.keys():
            individual_grad_concatenated += torch.flatten(local_model_updates[0][name]).tolist()
            cur_param = torch.cat([item[name].unsqueeze(0) for item in local_model_updates], axis=0)
            grad_aggregate_concatenated += torch.flatten(torch.mean(cur_param, axis=0)).tolist()

        sample_individual_grad_concatenated.append(individual_grad_concatenated)
        sample_grad_aggregate_concatenated.append(grad_aggregate_concatenated)

        # Update the global model
        for name in global_model_param.keys():
            cur_param = torch.cat([item[name].unsqueeze(0) for item in local_model_updates], axis=0)
            global_model_param[name] += torch.mean(cur_param, axis=0)
            # print(global_model_param[name])
        
parser = argparse.ArgumentParser()
parser.add_argument("--total-nodes", dest="total_nodes", type=int, default=50)
parser.add_argument("--subset", type=int, default=2)
parser.add_argument("--batch-size", dest="batch_size", type=int, default=32)
parser.add_argument("--trainTotalRounds", type=int, default=30)
parser.add_argument("--nEpochs", type=int, default=1)
parser.add_argument("--version", type=str, default="test")
parser.add_argument("--model", type=str, default="linear")
parser.add_argument("--dataset", type=str, default="mnist")
parser.add_argument("--lr", type=float, default=0.03)
args = parser.parse_args()

main(args=args)