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

        if iRound % args.interval == 0:
            # Start iteration of MINE
            sample_individual_grad_concatenated = []
            sample_grad_aggregate_concatenated = []
            local_model_updates = []
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
                # assert(len(local_model_updates) == nClients * num_jobs)
                individual_grad_concatenated = []
                grad_aggregate_concatenated = []

                for name in global_model_param.keys():
                    individual_grad_concatenated += torch.flatten(local_model_updates[0][name]).tolist()
                    cur_param = torch.cat([item[name].unsqueeze(0) for item in local_model_updates], axis=0)
                    grad_aggregate_concatenated += torch.flatten(torch.mean(cur_param, axis=0)).tolist()

                sample_individual_grad_concatenated.append(individual_grad_concatenated)
                sample_grad_aggregate_concatenated.append(grad_aggregate_concatenated)

            # # Train MINE network
            # X = np.array(sample_individual_grad_concatenated)
            # Y = np.array(sample_grad_aggregate_concatenated)
            # joint = torch.from_numpy(np.concatenate([X, Y], axis=1).astype("float32"))
            
            # for k in range(args.k):
            #     mine_net = Mine(input_size=grad_dim * 2).to(device)
            #     mine_net_optim = torch.optim.Adam(mine_net.parameters(), lr=0.01)
            #     mine_net.train()
            #     mine_results[iRound][k] = []
            #     random.shuffle(sample_grad_aggregate_concatenated)
            #     Y_ = np.array(sample_grad_aggregate_concatenated)
            #     margin = torch.from_numpy(np.concatenate([X, Y_], axis=1).astype("float32"))
            #     mine_dataset = MINEDataset(joint, margin)
            #     mine_traindataloader = DataLoader(mine_dataset, batch_size=args.mine_batch_size, shuffle=True)
                
            #     for niter in range(num_iter):
            #         mi_lb_sum = 0
            #         ma_et = 1
            #         for i, batch in enumerate(mine_traindataloader):
            #             mi_lb, ma_et = learn_mine(batch, device, mine_net, mine_net_optim, ma_et)
            #             mi_lb_sum += mi_lb
            #         if niter % 10 == 0:
            #             logger.info(f"MINE iter: {niter}, MI estimation: {mi_lb_sum / (i+1)}")
            #         mine_results[iRound][k].append((niter, mi_lb.item()))
                    
        else:
            local_model_updates = []
            # Get the global model
            ray.get([worker.pull_global_model.remote(global_model_param) for worker in workerByClient])

            # Run local training epochs
            ray.get([workerByClient[i].run_train_epoch.remote(
                ep=nEpochs, 
                update_global_state=True,
                traindataloader=dataloaderByClient[clientSubSet[i]]
                ) for i in range(len(workerByClient))
            ])

            # Get local model updates
            tmp_local_model_updates = ray.get([worker.push_local_model_updates.remote(global_model_param) for worker in workerByClient])
            local_model_updates += deepcopy(tmp_local_model_updates)    

        # Update the global model
        for name in global_model_param.keys():
            cur_param = torch.cat([item[name].unsqueeze(0) for item in local_model_updates], axis=0)
            global_model_param[name] += torch.mean(cur_param, axis=0)
            # print(global_model_param[name])

    # torch.save(mine_net, f"./param/mine_{subset}_{args.version}.bin")
    with open(f"./results/loss_{subset}_{args.version}.json", "w") as json_file:
        json.dump(mine_results, json_file)
        
parser = argparse.ArgumentParser()
parser.add_argument("--total-nodes", dest="total_nodes", type=int, default=50)
parser.add_argument("--subset", type=int, default=50)
parser.add_argument("--batch-size", dest="batch_size", type=int, default=32)
parser.add_argument("--mine-batch-size", dest="mine_batch_size", type=int, default=100)
parser.add_argument("--num-sample", dest="num_sample", type=int, default=3)
parser.add_argument("--trainTotalRounds", type=int, default=30)
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
args = parser.parse_args()

main(args=args)