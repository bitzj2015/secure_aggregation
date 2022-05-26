import os
import torch
import constant
import numpy as np
import argparse
from mine import *
from model import *
from dataset import *
import ray
import logging
import subprocess
from copy import deepcopy
torch.manual_seed(0)

# from utils.utils import label_to_onehot, cross_entropy_for_onehot
# from utils.utils import build_network
from utils.dataproc import data_lowrank, data_withnoise

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
        NUM_GPUS = 1
    else:
        NUM_GPUS = 0
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

    dataloaderByClient, testdataloader = get_dataset(args.dataset, batch_size, nClients, logger, sampling=args.sampling, alpha=args.alpha)
    
    if args.gpu_pw > 0:
        ray.init(num_gpus=NUM_GPUS) #, device=device
        constant.gpu_per_worker = NUM_GPUS / len(clientSubSet)
        constant.cpu_per_worker = 0
    else:
        ray.init(num_cpus=os.cpu_count()) #, device=device
        constant.gpu_per_worker = 0
        constant.cpu_per_worker = int(os.cpu_count() / len(clientSubSet))
    from worker import Worker

    workerByClient = [Worker.remote(dataloaderByClient[clientSubSet[i]], lr=args.lr, model_name=args.model, dataset_name=args.dataset, device=device, algo=args.algo) for i in range(len(clientSubSet))]
    global_model_param = ray.get(workerByClient[0].get_local_model_param.remote())
    mine_results = {}

    for iRound in range(trainTotalRounds):
        mine_results[iRound] = {}
        ray.get([worker.pull_global_model.remote(global_model_param) for worker in workerByClient])
        acc = ray.get(workerByClient[0].run_eval_epoch.remote(testdataloader))
        logger.info(f"Round: {iRound}, test accuracy: {acc}")

        local_model_updates = []
        for n in range(num_jobs):
            # Run local training epochs
            res = ray.get([worker.run_train_epoch.remote(ep=nEpochs, update_global_state=True) for worker in workerByClient])
            if n == 0:
                data_batch = res[0]

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

        # Update the global model
        for name in global_model_param.keys():
            cur_param = torch.cat([item[name].unsqueeze(0) for item in local_model_updates], axis=0)
            global_model_param[name] += torch.mean(cur_param, axis=0)

        # Train MINE network
        X = torch.Tensor(np.array(individual_grad_concatenated)).to(device)
        Y = torch.Tensor(np.array(grad_aggregate_concatenated)).to(device)
        (image, label) = data_batch
        # print(Y.shape, image.size(), label.size())
        ray.get(workerByClient[0].run_dlg.remote(image, label, Y, iRound))
        
        
parser = argparse.ArgumentParser()
parser.add_argument("--total-nodes", dest="total_nodes", type=int, default=50)
parser.add_argument("--subset", type=int, default=50)
parser.add_argument("--batch-size", dest="batch_size", type=int, default=32)
parser.add_argument("--mine-batch-size", dest="mine_batch_size", type=int, default=100)
parser.add_argument("--num-sample", dest="num_sample", type=int, default=10)
parser.add_argument("--trainTotalRounds", type=int, default=1)
parser.add_argument("--nEpochs", type=int, default=1)
parser.add_argument("--version", type=str, default="test")
parser.add_argument("--model", type=str, default="linear")
parser.add_argument("--dataset", type=str, default="mnist")
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.03)
parser.add_argument("--alpha", type=float, default=1)
parser.add_argument("--sampling", type=str, default="iid")
parser.add_argument("--algo", type=str, default="fedsgd")
parser.add_argument("--gpu-pw", dest="gpu_pw", type=float, default=0)
parser.add_argument("--interval", dest="interval", type=int, default=1)
args = parser.parse_args()

main(args=args)