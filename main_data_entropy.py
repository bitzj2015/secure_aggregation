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
import matplotlib.pyplot as plt

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


    num_iter = 1000
    num_samples = args.num_sample
    dataloaderByClient, testdataloader = get_dataset(args.dataset, batch_size, nClients, logger)
    np.random.seed(1)
    ray.init()

    mine_results = {}

    if args.resample:
        for k in range(10):
            mine_results[k] = {}
            all_sample_individual_grad_concatenated = []
            all_sample_grad_aggregate_concatenated = []
            xdata = []

            for _ in tqdm(range(num_samples)):
                # np.random.seed(0)
                clientSubSet = np.random.choice(nClients, subset, replace=True)
                # clientSubSet = [0] + [item + 1 for item in clientSubSet]
                
                workerByClient = [Worker.remote(dataloaderByClient[clientSubSet[i]], lr=args.lr, model_name=args.model, dataset_name=args.dataset) for i in range(len(clientSubSet))]
                global_model_param = ray.get(workerByClient[0].get_local_model_param.remote())
                # grad_dim = ray.get(workerByClient[0].get_grad_dim.remote())
                # print(clientSubSet[0])
                sample_individual_grad_concatenated = []
                sample_grad_aggregate_concatenated = []
                xdata.append(dataloaderByClient[clientSubSet[0]].dataset.input.cpu().numpy().reshape(-1))
                # print(xdata[-1].shape)
                for iRound in range(trainTotalRounds):
                    ray.get([worker.pull_global_model.remote(global_model_param) for worker in workerByClient])
                    acc = ray.get(workerByClient[0].run_eval_epoch.remote(testdataloader))
                    logger.info(f"Round: {iRound}, test accuracy: {acc}")

                    local_model_updates = ray.get([worker.push_local_model_updates.remote(global_model_param) for worker in workerByClient])
                    for name in global_model_param.keys():
                        cur_param = torch.cat([item[name].unsqueeze(0) for item in local_model_updates], axis=0)
                        global_model_param[name] += torch.mean(cur_param, axis=0)
                    
                    local_model_updates = []
                    for m in range(num_jobs):
                        # Get the global model
                        
                        ray.get([worker.pull_global_model.remote(global_model_param) for worker in workerByClient])

                        # Run local training epochs
                        for _ in range(nEpochs):
                            ray.get([worker.run_train_epoch.remote() for worker in workerByClient])

                        # Get local model updates
                        tmp_local_model_updates = ray.get([worker.push_local_model_updates.remote(global_model_param) for worker in workerByClient])
                        local_model_updates += deepcopy(tmp_local_model_updates)

                    individual_grad_concatenated = []
                    grad_aggregate_concatenated = []
                    grad_concatenated = [[] for _ in range(len(clientSubSet))]

                    for name in global_model_param.keys():
                        individual_grad_concatenated += torch.flatten(local_model_updates[0][name]).tolist()
                        for n in range(len(clientSubSet)):
                            grad_concatenated[n] += torch.flatten(local_model_updates[n][name]).tolist()
                        cur_param = torch.cat([item[name].unsqueeze(0) for item in local_model_updates], axis=0)
                        grad_aggregate_concatenated += torch.flatten(torch.mean(cur_param, axis=0)).tolist()

                    sample_individual_grad_concatenated.append(individual_grad_concatenated)
                    sample_grad_aggregate_concatenated.append(grad_aggregate_concatenated)
                    
                    for n in range(len(clientSubSet)):
                        plt.figure()
                        plt.plot(grad_concatenated[n][0:200], ".")
                        plt.savefig(f"./figs/test_{iRound}_{n}.jpg")

                    # Update the global model
                    for name in global_model_param.keys():
                        cur_param = torch.cat([item[name].unsqueeze(0) for item in local_model_updates], axis=0)
                        global_model_param[name] += torch.mean(cur_param, axis=0)

                all_sample_individual_grad_concatenated.append(deepcopy(sample_individual_grad_concatenated))
                all_sample_grad_aggregate_concatenated.append(deepcopy(sample_grad_aggregate_concatenated))

            X = np.array(xdata)
            Y1 = np.array([np.array(grad[:trainTotalRounds]).reshape(-1) for grad in all_sample_individual_grad_concatenated])
            Y2 = np.array([np.array(grad[:trainTotalRounds]).reshape(-1) for grad in all_sample_grad_aggregate_concatenated])

            hf = h5py.File(f"./dataset/grad_data_{subset}_{k}_{args.dataset}.hdf5", "w")
            hf.create_dataset('xdata', data=X)
            hf.create_dataset('ygrad1', data=Y1)
            hf.create_dataset('ygrad2', data=Y2)
            hf.close()
    else:
        mine_results = {}
        for k in range(10):
            mine_results[k] = {}
            hf = h5py.File(f"./dataset/grad_data_{subset}_{k}_{args.dataset}.hdf5", "r")
            X = np.array(hf["xdata"][:])
            X = X.reshape(X.shape[0] * 1000, -1)
            # X_all = list(np.array(hf["ygrad1"][:]))
            Y_all = list(X)
            print(X.shape, len(Y_all[0]))

            for iRound in [1]: # [1, 5, 10, 20, 30]:
                mine_results[k][iRound] = []
                # Train MINE network
                # X = np.array([np.array(grad[:iRound * 7850]).reshape(-1) for grad in X_all]) * 1000

                # Y = np.array([np.array(grad[:iRound * 7850]).reshape(-1) for grad in Y_all]) * 1000
                Y = np.array([np.array(grad).reshape(-1) for grad in Y_all])

                joint = torch.from_numpy(np.concatenate([X, Y], axis=1).astype("float32"))
                random.shuffle(Y_all)
                # Y_ = np.array([np.array(grad[:iRound * 7850]).reshape(-1) for grad in Y_all])
                Y_ = np.array([np.array(grad).reshape(-1) for grad in Y_all])

                margin = torch.from_numpy(np.concatenate([X, Y_], axis=1).astype("float32"))

                mine_dataset = MINEDataset(joint, margin)
                mine_traindataloader = DataLoader(mine_dataset, batch_size=args.mine_batch_size, shuffle=True)
                # print(X.shape, Y.shape, margin.size(), joint.size())
                mine_net = Mine(input_size=X.shape[1] + Y.shape[1]).to(device)
                mine_net_optim = torch.optim.Adam(mine_net.parameters(), lr=0.01)
                mine_net.train()

                for niter in range(num_iter):
                    mi_lb_sum = 0
                    ma_et = 1
                    for i, batch in enumerate(mine_traindataloader):
                        mi_lb, ma_et = learn_mine(batch, device, mine_net, mine_net_optim, ma_et)
                        mi_lb_sum += mi_lb
                    if niter % 10 == 0:
                        logger.info(f"MINE iter: {niter}, MI estimation: {mi_lb_sum / (i+1)}")
                    mine_results[k][iRound].append((niter, mi_lb.item()))

        # torch.save(mine_net, f"./param/mine_{subset}_{args.version}.bin")
        with open(f"./results/loss_{subset}_{args.version}.json", "w") as json_file:
            json.dump(mine_results, json_file)
        
parser = argparse.ArgumentParser()
parser.add_argument("--total-nodes", dest="total_nodes", type=int, default=50)
parser.add_argument("--subset", type=int, default=20)
parser.add_argument("--batch-size", dest="batch_size", type=int, default=256)
parser.add_argument("--mine-batch-size", dest="mine_batch_size", type=int, default=1)
parser.add_argument("--num-sample", dest="num_sample", type=int, default=1)
parser.add_argument("--trainTotalRounds", type=int, default=30)
parser.add_argument("--nEpochs", type=int, default=1)
parser.add_argument("--version", type=str, default="test")
parser.add_argument("--model", type=str, default="linear")
parser.add_argument("--dataset", type=str, default="mnist")
parser.add_argument("--lr", type=float, default=0.03)
parser.add_argument('--resample', dest="resample", default=False, action='store_true')
args = parser.parse_args()

main(args=args)