import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import random
import argparse
from torchvision import datasets
from torchvision.transforms import ToTensor
from mine import *
from model import *
import ray




@ray.remote
class Worker(object):
    def __init__(self, local_model, local_optimizer, local_dataset):
        self.local_model = local_model
        self.local_optimizer = local_optimizer
        self.local_dataset = local_dataset
    
    def run_train_epoch(self):
        return
    
    def pull_global_model(self, global_model):
        with torch.no_grad():
            for param, global_param in zip(self.local_model.parameters(), global_model.parameters()):
                param.data = global_param.clone().detach()

    def push_local_model(self):
        return



def main(args):
    train_data = datasets.MNIST(
        root = 'data',
        train = True,                         
        transform = ToTensor(), 
        download = True,            
    )
    test_data = datasets.MNIST(
        root = 'data', 
        train = False, 
        transform = ToTensor()
    )

    nClients = args.total_nodes
    trainTotalRounds = args.trainTotalRounds
    Batch_size = args.Batch_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    # device = 'cpu'


    seed = 2
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    xtr = train_data.data / 255.0
    xtr = (xtr - 0.5) / 0.5
    ytr = train_data.targets
    xtr = xtr.to(device)
    ytr = ytr.to(device)
    trainDataSize = xtr.size(0)
    print(trainDataSize)


    xte = test_data.data / 255.0
    xte = (xte - 0.5) / 0.5
    yte = test_data.targets
    xte = xte.to(device)
    yte = yte.to(device)


    trainDataSizeFracClients = 1/nClients

    trainDataSizeClients = np.int32(trainDataSizeFracClients * trainDataSize)


    # time.sleep(3)
    Epochs = 1
    miniBatchSizeFrac = np.array([1 for iClient in range(nClients)])
    miniBatchSizeClients = np.int32(miniBatchSizeFrac * Batch_size)

    clientsEpochs = np.array([Epochs for iClient in range(nClients)])
    nMiniBatchesClientsPerEpoch = np.int32(trainDataSizeClients / miniBatchSizeClients)



    trainDataIndices = {}
    stIndex = 0
    for iClient in range(nClients):
        #print('iClient', iClient)
        #print('index',stIndex)
        #print('train data size', trainDataSizeClients)
        trainDataIndices[(iClient, 0)] = stIndex
        trainDataIndices[(iClient, 1)] = (stIndex + trainDataSizeClients)
        stIndex = (stIndex + trainDataSizeClients)
    # testBatchSize = 500
    # testTotalBatches = testDataSize/testBatchSize
    # assert (testTotalBatches-int(testTotalBatches)) == 0, "Non-integer issue"
    # testTotalBatches = int(testTotalBatches)

    lrIni = 0.03
    lr = lrIni





    net = Net()
    print(net)
    net = net.to(device)
    # print(net)
    # print("Total trainable parameters:", get_n_params(net))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)


    test_accuracy = []
    modelParamsClientsOld = {}
    modelParamsClientsNew = {}
    Grad_aggregate= {}
    GradssClients = {}
    Grad_aggregate_concatenated  ={} # concatenation of all layers

    individual_grad_concatenated = {}
    individual_grad = {}

    Grad_aggregate_flatten = {}

    subset = args.subset
    num_iter = 1000
    num_samples = 1000
    sub_set = np.random.choice(nClients-1, subset-1, replace=False)
    sub_set = [0] + [item + 1 for item in sub_set]

    print(net, sub_set)
    lcount = 0

    for param in net.parameters():
        for iClient in sub_set:
            modelParamsClientsOld[(iClient, lcount)] = param.clone().detach()  # at the end of previous round
            modelParamsClientsNew[(iClient, lcount)] = param.clone().detach()
        lcount = lcount + 1

    # mine_net = Mine(input_size=7850*2).to(device)
    # mine_net_optim = optim.Adam(mine_net.parameters(), lr=1e-3)
    # mine_net.train()
    mine_results = {}

    for iRound in range(trainTotalRounds):
        mine_net = Mine(input_size=7850*2).to(device)
        mine_net_optim = optim.Adam(mine_net.parameters(), lr=1e-3)
        mine_net.train()
        
        mine_results[iRound] = []
        lr= lrIni
        lr = .03
        a = sub_set[0]
        lcount = 0
        for param in net.parameters():
            modelParamsClientsNew[(a, lcount)].fill_(0)
            for client_i in sub_set:
                modelParamsClientsNew[(a, lcount)] += 1 / subset * modelParamsClientsOld[(client_i, lcount)]
            # print(modelParamsClientsOld[(client_i, lcount)])
            lcount = lcount + 1
        # print(modelParamsClientsNew[(a, 0)])

        with torch.no_grad():
            lcount = 0
            for name, param in net.named_parameters():
                param.data = modelParamsClientsNew[(a, lcount)].clone().detach()
                print(name)
                #print(param.size())
                lcount = lcount + 1

        with torch.no_grad():
            net.eval()
            xTestBatch = xte
            yTestBatch = yte
            outputs = net(xTestBatch)
            _, predicted_val = torch.max(outputs.data, 1)
            total = yTestBatch.size(0)
            correct = (predicted_val == yTestBatch).sum().item()
            accIter = correct / total

        test_accuracy.append(accIter)
        print('Round:', (iRound + 1), 'test accuracy:', accIter)


        # Start iteration of MINE
        
        sample_individual_grad_concatenated = [[] for _ in range(nClients)]
        sample_Grad_aggregate_concatenated = []
        # Sample gradients, i.e., repeat the FL training round for num_samples times from the same initial model
        for i in tqdm(range(num_samples)):
            for iClient in sub_set:
                with torch.no_grad():
                    lcount = 0
                    for param in net.parameters():
                        param.data = modelParamsClientsNew[(a, lcount)].clone().detach()
                        lcount = lcount + 1
                clientDataX = xtr[trainDataIndices[(iClient, 0)]:trainDataIndices[(iClient, 1)]]
                clientDataY = ytr[trainDataIndices[(iClient, 0)]:trainDataIndices[(iClient, 1)]]
                seed = iRound * nClients + iClient + i # + niter * num_samples # change random seed for each sampling round
                np.random.seed(seed)
                random.seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                
                nEpochs = clientsEpochs[iClient]
                # print(nEpochs,'nEpochs')
                nBatches = nMiniBatchesClientsPerEpoch[iClient]
                #print(nBatches,'nBatches')
                miniBatchSize = miniBatchSizeClients[iClient]
                for iEpoch in range(nEpochs):
                    batch_list = list(range(nBatches))
                    random.shuffle(batch_list) # shuffing batch such that each sampling round will generate different gradients
                    for iTrainMiniBatch in batch_list:
                        xbatch = clientDataX[iTrainMiniBatch * miniBatchSize:(iTrainMiniBatch + 1) * miniBatchSize]
                        ybatch = clientDataY[iTrainMiniBatch * miniBatchSize:(iTrainMiniBatch + 1) * miniBatchSize]

                        net.train()
                        optimizer.zero_grad()
                        outputs = net(xbatch)
                        loss = criterion(outputs, ybatch)
                        loss.backward()
                        
                        with torch.no_grad():
                            lcount = 0
                            for param in net.parameters():
                                gradParams = param.grad.clone().detach()
                                # print(gradParams.size(), param.grad)
                                
                                param.data -= lr * gradParams
                                GradssClients[((iClient, lcount))] = gradParams
                                modelParamsClientsOld[(iClient, lcount)] = param.data
                                lcount = lcount + 1
                
                lcount = 0
                for param in net.parameters():
                    GradssClients[((iClient, lcount))] = modelParamsClientsOld[(iClient, lcount)]  -  modelParamsClientsNew[(a, lcount)]
                    lcount = lcount + 1

            lrIni = lrIni*1
            lcount = 0

            for param in net.parameters():
                Grad_aggregate[(0, lcount)]= torch.zeros_like(GradssClients[((a, lcount))])
                for client_i in sub_set:
                    Grad_aggregate[(0, lcount)] +=  GradssClients[((client_i, lcount))]
                Grad_aggregate[(0, lcount)] /= len(sub_set)
                lcount = lcount + 1
            # print(Grad_aggregate[(0, 0)].shape)


            for iClient in sub_set:  # The average old model will be copied for each client
                lcount = 0
                Grad_aggregate_concatenated = []

                individual_grad_concatenated [iClient]= []
                for param in net.parameters():
                    Grad_aggregate_flatten[(0, lcount)] = torch.flatten(Grad_aggregate[(0, lcount)]).tolist()
                    Grad_aggregate_concatenated = Grad_aggregate_concatenated  + Grad_aggregate_flatten[(0, lcount)]

                    individual_grad[(iClient, lcount)] = torch.flatten(GradssClients[(iClient, lcount)]).tolist()
                    individual_grad_concatenated[iClient] = individual_grad_concatenated[iClient]  + individual_grad[(iClient, lcount)]
                    lcount = lcount + 1
                
            ## compute the mutual information 
            for iClient in sub_set: 
                sample_individual_grad_concatenated[iClient].append(individual_grad_concatenated[iClient])
            sample_Grad_aggregate_concatenated.append(Grad_aggregate_concatenated)

        # print(len(sample_individual_grad_concatenated[iClient]), len(sample_Grad_aggregate_concatenated[0]))

        X = np.array(sample_individual_grad_concatenated[0])
        Y = np.array(sample_Grad_aggregate_concatenated)
        sample_Grad_aggregate_concatenated.reverse()
        Y_ = np.array(sample_Grad_aggregate_concatenated)
        for niter in range(num_iter):
            joint = np.concatenate([X, Y], axis=1)
            margin = np.concatenate([X, Y_], axis=1)

            batch = torch.from_numpy(joint).float(), torch.from_numpy(margin).float()
            ma_et = 1
            mi_lb, ma_et = learn_mine(batch, mine_net, mine_net_optim, ma_et)
            print(f"MINE iter: {niter}, MI estimation: {mi_lb}")
            mine_results[iRound].append((niter, mi_lb.item()))


        # ## compute the mutual information 
        # tmp = []
        # # print(sample_individual_grad_concatenated[iClient][0][0:10])
        # # print(sample_individual_grad_concatenated[iClient][1][0:10])
        # # print(sample_Grad_aggregate_concatenated[0][0:10])
        # for j in range(len(sample_Grad_aggregate_concatenated[0])):
        #     for iClient in sub_set: 
        #         MI_with_iClient[(iClient,iRound)]= MI.mi_Kraskov([
        #             [item[j] for item in sample_individual_grad_concatenated[iClient]],
        #             [item[j] for item in sample_Grad_aggregate_concatenated]
        #         ],k=5,base=np.exp(1)) 
        #         tmp.append(MI_with_iClient[(iClient,iRound)])
        #     break
        # print(np.mean(tmp), np.min(tmp), np.max(tmp))
        # Mutual_information_with_iclient.append([np.mean(tmp), np.min(tmp), np.max(tmp)])
    # Mutual_information_with_iclient = np.array(Mutual_information_with_iclient)
    # np.save('Mutual_information_Number_clients_MNIST_{}'.format(subset),  MI_with_iClient)
    #print(MI_with_iClient)
    # print('Number_clients', nClients)
    torch.save(mine_net, f"./param/mine_{subset}.bin")
    with open(f"./param/loss_{subset}.json", "w") as json_file:
        json.dump(mine_results, json_file)
        
parser = argparse.ArgumentParser()
parser.add_argument("--total_nodes", type=int, default=100)
parser.add_argument("--subset", type=int, default=10)
parser.add_argument("--Batch_size", type=int, default=32)
parser.add_argument("--trainTotalRounds", type=int, default=30)


# parser.add_argument("--connet_pro", type=float,default=0.4)
args = parser.parse_args()

main(args=args)