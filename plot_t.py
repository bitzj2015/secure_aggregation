import json
from pickle import TRUE
from tkinter import VERTICAL
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.rcParams['font.family'] = 'Times New Roman'
import numpy as np

num_user = 20
# entropy = 1403 * 1000 # 25088 
# entropy = 567 * 32 # 1924
FONTSIZE = 24

z_conf = {"80": 1.28, "90": 1.645, "95": 1.96, "98": 2.33, "99": 2.58}
conf = "95"
fig_location = "fedprox/final"
tag = "_fedprox"
dataset = "cifar10"
K = 3

# fig, ax = plt.subplots()
# ax.yaxis.set_major_locator(plt.MaxNLocator(6))
# ax.xaxis.set_major_locator(plt.MaxNLocator(6))
# ax.tick_params(axis='x', labelsize=FONTSIZE)
# ax.tick_params(axis='y', labelsize=FONTSIZE)
# R = [1, 5, 10, 20, 30]

# for version in ["mnist_may_fedprox", "cifar10_may_fedprox"]:
# for version in ["lin_20_avg_t2", "lin_20_avg_t_cifar10"]:
#     with open(f"./results/lin/loss_{num_user}_{version}.json", "r") as json_file:
#     # with open(f"./results/fedprox/linear/loss_{version}.json", "r") as json_file:
#         data = json.load(json_file)

#     for use_norm in ["low"]:
#         all_mi = {}
#         all_mi_min = {}
#         all_mi_max = {}
#         for iRound in R:
#             mi = []
#             for k in range(K):
#                 res = []
#                 for item in data[str(k)][str(iRound)]:
#                     if item[1] != float("inf") and item[1] != float("nan"):
#                         res.append(item[1])
#                 mi.append(max(res))
#             if "cifar10" in version:
#                 entropy = 1403 * 1000
#             else:
#                 entropy = 567 * 1200
#             if use_norm == "low":
#                 mi = [item * 100 / entropy  for item in mi]
#                 all_mi[iRound] = np.mean(mi)
#                 all_mi_min[iRound] = np.mean(mi) - z_conf[conf] * np.std(mi) / np.sqrt(len(mi)) 
#                 all_mi_max[iRound] = np.mean(mi) + z_conf[conf] * np.std(mi) / np.sqrt(len(mi))
#             else:
#                 all_mi[iRound] = np.mean(mi)
#                 all_mi_min[iRound] = np.mean(mi) - z_conf[conf] * np.std(mi) / np.sqrt(len(mi)) 
#                 all_mi_max[iRound] = np.mean(mi) + z_conf[conf] * np.std(mi) / np.sqrt(len(mi))

#         ax.plot(list(all_mi.keys()), list(all_mi.values()), "*-")
#         ax.fill_between(R, list(all_mi_min.values()), list(all_mi_max.values()), alpha=.1)
# ax.set_xlabel("Training round T", fontsize=FONTSIZE)
# ax.grid(True)
# if use_norm == "low":
#     ax.set_ylabel("Normalized MI (%)", fontsize=FONTSIZE)
#     ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# elif use_norm == "high":
#     ax.set_ylabel("Unnormalized MI (bits)", fontsize=FONTSIZE)
# ax.legend(['MNIST',"CIFAR10"], fontsize=22, ncol=2)
# fig.savefig(f"./results/{fig_location}/results_{version}_{use_norm}{tag}.jpg", bbox_inches='tight')

acc_all = {}
for algo in ["fedsgd", "fedavg", "fedprox"]:
    acc_all[algo] = []
    filename = f"./logs/log_may_{dataset}_{algo}.txt"
    with open(filename, "r") as file:
        cnt = 0
        for line in file.readlines():
            cnt += 1
            if cnt <= 2:
                continue
            item = line.replace('\n','').split(" ")
            acc_all[algo].append(float(item[-1]) * 100)

fig, ax = plt.subplots()
ax.yaxis.set_major_locator(plt.MaxNLocator(6))
ax.xaxis.set_major_locator(plt.MaxNLocator(6))
ax.tick_params(axis='x', labelsize=FONTSIZE)
ax.tick_params(axis='y', labelsize=FONTSIZE)

all_mi = {}
all_mi_r = {}
all_mi_min = {}
all_mi_max = {}
R = [50, 45, 40, 35, 30, 25, 20, 15, 10, 5]

for version in R:
    with open(f"./results/fedprox/accum/loss_{dataset}_may_fedsgd_{version}.json", "r") as json_file:
        data = json.load(json_file)

    for iRound in [version - item for item in range(5)]:
        mi = []
        for k in range(K):
            res = []
            for item in data[str(k)][str(iRound)]:
                if item[1] != float("inf") and item[1] != float("nan"):
                    res.append(item[1])
            mi.append(max(res))
        # if "cifar10" in version:
        #     entropy = 1403 * 1000
        # else:
        entropy = 567 * 1200
        mi = [item * 100 / entropy  for item in mi]
        all_mi_r[iRound] = mi
    
for iRound in range(1,50,1):

    for k in range(K):
        if all_mi_r[iRound+1][k] < all_mi_r[iRound][k]:
            all_mi_r[iRound+1][k] = all_mi_r[iRound][k]
    mi = all_mi_r[iRound]
    all_mi[iRound] = np.mean(mi)
    all_mi_min[iRound] = np.mean(mi) - z_conf[conf] * np.std(mi) / np.sqrt(len(mi)) 
    all_mi_max[iRound] = np.mean(mi) + z_conf[conf] * np.std(mi) / np.sqrt(len(mi))

iRound += 1
mi = all_mi_r[iRound]
all_mi[iRound] = np.mean(mi)
all_mi_min[iRound] = np.mean(mi) - z_conf[conf] * np.std(mi) / np.sqrt(len(mi)) 
all_mi_max[iRound] = np.mean(mi) + z_conf[conf] * np.std(mi) / np.sqrt(len(mi))

all_mi = dict(sorted(all_mi.items()))
ax.plot(list(all_mi.keys())[:10], list(all_mi.values())[:10], "*-")
ax.fill_between(list(range(50))[:10], list(all_mi_min.values())[:10], list(all_mi_max.values())[:10], alpha=.1)

all_mi_r = {}
all_mi = {}
all_mi_min = {}
all_mi_max = {}
R = [10, 5]

for version in R:
    with open(f"./results/fedprox/accum/loss_{dataset}_may_fedavg_{version}.json", "r") as json_file:
        data = json.load(json_file)
    for iRound in [version - item for item in range(5)]:
        mi = []
        for k in range(K):
            res = []
            for item in data[str(k)][str(iRound)]:
                if item[1] != float("inf") and item[1] != float("nan"):
                    res.append(item[1])
            mi.append(max(res))
        # if "cifar10" in version:
        #     entropy = 1403 * 1000
        # else:
        entropy = 567 * 1200
        mi = [item * 100 / entropy  for item in mi]
        all_mi_r[iRound] = mi

for iRound in range(1,R[0],1):

    for k in range(K):
        if all_mi_r[iRound+1][k] < all_mi_r[iRound][k]:
            all_mi_r[iRound+1][k] = all_mi_r[iRound][k]
    mi = all_mi_r[iRound]
    all_mi[iRound] = np.mean(mi)
    all_mi_min[iRound] = np.mean(mi) - z_conf[conf] * np.std(mi) / np.sqrt(len(mi)) 
    all_mi_max[iRound] = np.mean(mi) + z_conf[conf] * np.std(mi) / np.sqrt(len(mi))

iRound += 1
mi = all_mi_r[iRound]
all_mi[iRound] = np.mean(mi)
all_mi_min[iRound] = np.mean(mi) - z_conf[conf] * np.std(mi) / np.sqrt(len(mi)) 
all_mi_max[iRound] = np.mean(mi) + z_conf[conf] * np.std(mi) / np.sqrt(len(mi))

ax.plot(list(all_mi.keys())[:10], list(all_mi.values())[:10], "*-")
ax.fill_between(list(range(30))[:10], list(all_mi_min.values())[:10], list(all_mi_max.values())[:10], alpha=.1)

all_mi_r = {}
all_mi = {}
all_mi_min = {}
all_mi_max = {}

for version in R:
    with open(f"./results/fedprox/accum/loss_{dataset}_may_fedprox_{version}.json", "r") as json_file:
        data = json.load(json_file)

    for iRound in [version - item for item in range(5)]:
        mi = []
        for k in range(K):
            res = []
            for item in data[str(k)][str(iRound)]:
                if item[1] != float("inf") and item[1] != float("nan"):
                    res.append(item[1])
            mi.append(max(res))
        # if "cifar10" in version:
        #     entropy = 1403 * 1000
        # else:
        entropy = 567 * 1200
        mi = [item * 100 / entropy  for item in mi]
        all_mi_r[iRound] = mi

for iRound in range(1,R[0],1):

    for k in range(K):
        if all_mi_r[iRound+1][k] < all_mi_r[iRound][k]:
            all_mi_r[iRound+1][k] = all_mi_r[iRound][k]
    mi = all_mi_r[iRound]
    all_mi[iRound] = np.mean(mi)
    all_mi_min[iRound] = np.mean(mi) - z_conf[conf] * np.std(mi) / np.sqrt(len(mi)) 
    all_mi_max[iRound] = np.mean(mi) + z_conf[conf] * np.std(mi) / np.sqrt(len(mi))

iRound += 1
mi = all_mi_r[iRound]
all_mi[iRound] = np.mean(mi)
all_mi_min[iRound] = np.mean(mi) - z_conf[conf] * np.std(mi) / np.sqrt(len(mi)) 
all_mi_max[iRound] = np.mean(mi) + z_conf[conf] * np.std(mi) / np.sqrt(len(mi))

ax.plot(list(all_mi.keys())[:10], list(all_mi.values())[:10], "*-")
ax.fill_between(list(range(30))[:10], list(all_mi_min.values())[:10], list(all_mi_max.values())[:10], alpha=.1)

ax.set_xlabel("Training round T", fontsize=FONTSIZE)
ax.grid(True)
ax.set_ylabel("Normalized MI (%)", fontsize=FONTSIZE)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.legend(["FedSGD", "FedAvg", "FedProx"], fontsize=FONTSIZE * 0.8)
fig.savefig(f"./results/{fig_location}/results_accum_all_{dataset}.jpg", bbox_inches='tight')


######################################
fig, ax = plt.subplots()
ax.yaxis.set_major_locator(plt.MaxNLocator(6))
ax.xaxis.set_major_locator(plt.MaxNLocator(6))
ax.tick_params(axis='x', labelsize=FONTSIZE)
ax.tick_params(axis='y', labelsize=FONTSIZE)

all_mi = {}
all_mi_r = {}
all_mi_min = {}
all_mi_max = {}
R = [50, 45, 40, 35, 30, 25, 20, 15, 10, 5]
LEN  = 30

for version in R:
    with open(f"./results/fedprox/accum/loss_{dataset}_may_fedsgd_{version}.json", "r") as json_file:
        data = json.load(json_file)

    for iRound in [version - item for item in range(5)]:
        mi = []
        for k in range(K):
            res = []
            for item in data[str(k)][str(iRound)]:
                if item[1] != float("inf") and item[1] != float("nan"):
                    res.append(item[1])
            mi.append(max(res))
        # if "cifar10" in version:
        #     entropy = 1403 * 1000
        # else:
        entropy = 567 * 1200
        mi = [item * 100 / entropy  for item in mi]
        all_mi_r[iRound] = mi
    
for iRound in range(1,50,1):

    for k in range(K):
        if all_mi_r[iRound+1][k] < all_mi_r[iRound][k]:
            all_mi_r[iRound+1][k] = all_mi_r[iRound][k]
    mi = all_mi_r[iRound]
    all_mi[iRound] = np.mean(mi)
    all_mi_min[iRound] = np.mean(mi) - z_conf[conf] * np.std(mi) / np.sqrt(len(mi)) 
    all_mi_max[iRound] = np.mean(mi) + z_conf[conf] * np.std(mi) / np.sqrt(len(mi))

iRound += 1
mi = all_mi_r[iRound]
all_mi[iRound] = np.mean(mi)
all_mi_min[iRound] = np.mean(mi) - z_conf[conf] * np.std(mi) / np.sqrt(len(mi)) 
all_mi_max[iRound] = np.mean(mi) + z_conf[conf] * np.std(mi) / np.sqrt(len(mi))

all_mi = dict(sorted(all_mi.items()))
ax.plot(acc_all["fedsgd"][:LEN], list(all_mi.values())[:LEN], "*-")
ax.fill_between(acc_all["fedsgd"][:LEN], list(all_mi_min.values())[:LEN], list(all_mi_max.values())[:LEN], alpha=.1)
for i in range(len(acc_all["fedsgd"][:50])):
    if acc_all["fedsgd"][i] > 85:
        print("fedsgd", list(all_mi.values())[i])
        break

all_mi_r = {}
all_mi = {}
all_mi_min = {}
all_mi_max = {}
R = [30, 25, 20, 15, 10, 5]
LEN = 10
for version in R:
    with open(f"./results/fedprox/accum/loss_{dataset}_may_fedavg_{version}.json", "r") as json_file:
        data = json.load(json_file)

    for iRound in [version - item for item in range(5)]:
        mi = []
        for k in range(K):
            res = []
            for item in data[str(k)][str(iRound)]:
                if item[1] != float("inf") and item[1] != float("nan"):
                    res.append(item[1])
            mi.append(max(res))
        # if "cifar10" in version:
        #     entropy = 1403 * 1000
        # else:
        entropy = 567 * 1200
        mi = [item * 100 / entropy  for item in mi]
        all_mi_r[iRound] = mi

for iRound in range(1,R[0],1):

    for k in range(K):
        if all_mi_r[iRound+1][k] < all_mi_r[iRound][k]:
            all_mi_r[iRound+1][k] = all_mi_r[iRound][k]
    mi = all_mi_r[iRound]
    all_mi[iRound] = np.mean(mi)
    all_mi_min[iRound] = np.mean(mi) - z_conf[conf] * np.std(mi) / np.sqrt(len(mi)) 
    all_mi_max[iRound] = np.mean(mi) + z_conf[conf] * np.std(mi) / np.sqrt(len(mi))

iRound += 1
mi = all_mi_r[iRound]
all_mi[iRound] = np.mean(mi)
all_mi_min[iRound] = np.mean(mi) - z_conf[conf] * np.std(mi) / np.sqrt(len(mi)) 
all_mi_max[iRound] = np.mean(mi) + z_conf[conf] * np.std(mi) / np.sqrt(len(mi))

ax.plot(acc_all["fedavg"][:LEN], list(all_mi.values())[:LEN], "*-")
ax.fill_between(acc_all["fedavg"][:LEN], list(all_mi_min.values())[:LEN], list(all_mi_max.values())[:LEN], alpha=.1)
for i in range(len(acc_all["fedavg"][:LEN])):
    if acc_all["fedavg"][i] > 85:
        print("fedavg", list(all_mi.values())[i])
        break

all_mi_r = {}
all_mi = {}
all_mi_min = {}
all_mi_max = {}

for version in R:
    with open(f"./results/fedprox/accum/loss_{dataset}_may_fedprox_{version}.json", "r") as json_file:
        data = json.load(json_file)

    for iRound in [version - item for item in range(5)]:
        mi = []
        for k in range(K):
            res = []
            for item in data[str(k)][str(iRound)]:
                if item[1] != float("inf") and item[1] != float("nan"):
                    res.append(item[1])
            mi.append(max(res))
        # if "cifar10" in version:
        #     entropy = 1403 * 1000
        # else:
        entropy = 567 * 1200
        mi = [item * 100 / entropy  for item in mi]
        all_mi_r[iRound] = mi

for iRound in range(1,R[0],1):

    for k in range(K):
        if all_mi_r[iRound+1][k] < all_mi_r[iRound][k]:
            all_mi_r[iRound+1][k] = all_mi_r[iRound][k]
    mi = all_mi_r[iRound]
    all_mi[iRound] = np.mean(mi)
    all_mi_min[iRound] = np.mean(mi) - z_conf[conf] * np.std(mi) / np.sqrt(len(mi)) 
    all_mi_max[iRound] = np.mean(mi) + z_conf[conf] * np.std(mi) / np.sqrt(len(mi))

iRound += 1
mi = all_mi_r[iRound]
all_mi[iRound] = np.mean(mi)
all_mi_min[iRound] = np.mean(mi) - z_conf[conf] * np.std(mi) / np.sqrt(len(mi)) 
all_mi_max[iRound] = np.mean(mi) + z_conf[conf] * np.std(mi) / np.sqrt(len(mi))

ax.plot(acc_all["fedprox"][:LEN], list(all_mi.values())[:LEN], "*-")
ax.fill_between(acc_all["fedprox"][:LEN], list(all_mi_min.values())[:LEN], list(all_mi_max.values())[:LEN], alpha=.1)
for i in range(len(acc_all["fedprox"][:LEN])):
    if acc_all["fedprox"][i] > 85:
        print("fedprox", list(all_mi.values())[i])
        break


ax.set_xlabel("Model accuracy (%)", fontsize=FONTSIZE)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.grid(True)
ax.set_ylabel("Normalized MI (%)", fontsize=FONTSIZE)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.legend(["FedSGD", "FedAvg", "FedProx"], fontsize=FONTSIZE*0.8)
fig.savefig(f"./results/{fig_location}/results_accum_all2_{dataset}.jpg", bbox_inches='tight')