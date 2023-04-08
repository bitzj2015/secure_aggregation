import os
import torch
import random
import constant
import numpy as np
from tqdm import tqdm
import json
import argparse
from mine import *
from model import *
from dataset import *
import ray
import logging
import subprocess
from copy import deepcopy
torch.manual_seed(0)
BLOCK_SIZE = 500

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


    batch_size = args.batch_size
    trainTotalRounds = args.trainTotalRounds
    nEpochs = args.nEpochs
    subset = args.subset
    num_jobs = 1

    if args.gpu_pw > 0:
        ray.init(num_gpus=NUM_GPUS) #, device=device
        constant.cpu_per_worker = int(os.cpu_count() / subset)
        constant.gpu_per_worker = NUM_GPUS / subset
        constant.cpu_per_worker = 0
        worker_device = device
    else:
        ray.init(num_cpus=os.cpu_count()) #, device=device
        constant.gpu_per_worker = 0
        constant.cpu_per_worker = int(os.cpu_count() / subset)
        worker_device = device = torch.device("cpu")
    from worker import Worker

    dataloaderByClient, testdataloader, nClients = get_femnist(batch_size, logger)

    num_iter = 1000
    num_samples = args.num_sample
    np.random.seed(0)
    clientSubSet = np.random.choice(nClients-1, subset-1, replace=True)
    clientSubSet = [0] + [item + 1 for item in clientSubSet]

    workerByClient = [Worker.remote(
        dataloaderByClient[clientSubSet[i]], 
        lr=args.lr, 
        model_name=args.model, 
        dataset_name=args.dataset, 
        device=worker_device, 
        algo=args.algo, 
        num_class=62
        ) for i in range(len(clientSubSet))
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
        
        clientSubSet = np.random.choice(nClients-1, subset-1, replace=True)
        clientSubSet = [0] + [item + 1 for item in clientSubSet]

        # Start iteration of MINE
        sample_individual_grad_concatenated = [[] for _ in range(subset)]
    
        for m in tqdm(range(num_samples)):
            local_model_updates = []
            for _ in range(num_jobs):
                # Get the global model
                if m == num_samples - 1:
                    # Run local training epochs
                    ray.get([workerByClient[i].run_train_epoch.remote(
                        ep=nEpochs, 
                        update_global_state=True, 
                        traindataloader=dataloaderByClient[clientSubSet[i]]
                        ) for i in range(len(workerByClient))
                    ])
                else:
                    # Run local training epochs
                    ray.get([workerByClient[i].run_train_epoch.remote(
                        ep=nEpochs, 
                        update_global_state=False,
                        traindataloader=dataloaderByClient[clientSubSet[i]]
                        ) for i in range(len(workerByClient))
                    ])

                # Get local model updates
                tmp_local_model_updates = ray.get([worker.push_local_model_updates.remote(global_model_param) for worker in workerByClient])
                local_model_updates += deepcopy(tmp_local_model_updates)

            individual_grad_concatenated = [[] for _ in range(subset)]

            for user_id in range(subset):
                for name in global_model_param.keys():
                    individual_grad_concatenated[user_id] += torch.flatten(local_model_updates[user_id][name]).tolist()

                sample_individual_grad_concatenated[user_id].append(individual_grad_concatenated[user_id])

        # Train MINE network
        gradients = torch.from_numpy(np.array(sample_individual_grad_concatenated)).float()
        print(gradients.size())
        gradient_mean = gradients.mean(1).reshape(subset, 1, -1)
        gradients -= gradient_mean
        step = gradients.shape[-1] // BLOCK_SIZE + 1

        g_noise = [[] for _ in range(subset)]
        for t in range(step):
            sub_gradients = gradients[:, :, t * BLOCK_SIZE : t * BLOCK_SIZE + BLOCK_SIZE]
            sub_grad_dim = sub_gradients.size(-1)
            cov_mat = sub_gradients.permute(0,2,1).matmul(sub_gradients) / num_samples
            u, s, _ = torch.svd(cov_mat)
            s = torch.sqrt(s)
            for user_id in range(subset):
                orth_mat = u[user_id].matmul(torch.diag(s[user_id] + args.sigma))
                mean = torch.zeros(size=(sub_grad_dim,))
                mvn = torch.distributions.MultivariateNormal(mean, orth_mat.matmul(orth_mat.t()))
                sub_g_noise = mvn.sample()
                g_noise[user_id].append(sub_g_noise)
        a = torch.cat(g_noise[user_id], dim=0).reshape(-1) + gradient_mean[user_id].reshape(-1)
        g_noise = torch.cat([torch.cat(g_noise[user_id], dim=0).reshape(1,-1) + gradient_mean[user_id].reshape(1,-1) for user_id in range(subset)], dim=0)
        print(g_noise.size())

        # Update the global model
        start_dim = 0
        end_dim = start_dim
        for name in global_model_param.keys():
            end_dim += local_model_updates[0][name].reshape(-1).size(0)
            global_model_param[name] += g_noise[:, start_dim: end_dim].mean(0).reshape(local_model_updates[0][name].size())
            start_dim = end_dim

        
parser = argparse.ArgumentParser()
parser.add_argument("--total-nodes", dest="total_nodes", type=int, default=50)
parser.add_argument("--subset", type=int, default=50)
parser.add_argument("--batch-size", dest="batch_size", type=int, default=32)
parser.add_argument("--num-sample", dest="num_sample", type=int, default=100)
parser.add_argument("--trainTotalRounds", type=int, default=50)
parser.add_argument("--nEpochs", type=int, default=1)
parser.add_argument("--version", type=str, default="test")
parser.add_argument("--model", type=str, default="linear")
parser.add_argument("--dataset", type=str, default="mnist")
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--alpha", type=float, default=1)
parser.add_argument("--sampling", type=str, default="iid")
parser.add_argument("--algo", type=str, default="fedavg")
parser.add_argument("--gpu-pw", dest="gpu_pw", type=float, default=0)
parser.add_argument("--interval", dest="interval", type=int, default=1)
parser.add_argument("--sigma", dest="sigma", type=int, default=0.01)
args = parser.parse_args()

main(args=args)