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
fig_location = "final"


fig, ax = plt.subplots()
ax.yaxis.set_major_locator(plt.MaxNLocator(6))
ax.xaxis.set_major_locator(plt.MaxNLocator(6))
ax.tick_params(axis='x', labelsize=FONTSIZE)
ax.tick_params(axis='y', labelsize=FONTSIZE)

for version in ["lin_20_avg_t_mnist", "lin_20_avg_t_cifar10_sgd"]:
# for version in ["lin_20_avg_t2", "lin_20_avg_t_cifar10"]:
    with open(f"./results/lin/loss_{num_user}_{version}.json", "r") as json_file:
        data = json.load(json_file)

    for use_norm in ["low"]:
        all_mi = {}
        all_mi_min = {}
        all_mi_max = {}
        for iRound in [1, 5, 10, 20, 30]:
            mi = []
            for k in range(5):
                res = []
                for item in data[str(k)][str(iRound)]:
                    if item[1] != float("inf") and item[1] != float("nan"):
                        res.append(item[1])
                mi.append(max(res))
            if "cifar10" in version:
                entropy = 1403 * 1000
            else:
                entropy = 567 * 1200
            if use_norm == "low":
                mi = [item / entropy * 100 for item in mi]
                all_mi[iRound] = np.mean(mi)
                all_mi_min[iRound] = np.mean(mi) - z_conf[conf] * np.std(mi) / np.sqrt(len(mi)) 
                all_mi_max[iRound] = np.mean(mi) + z_conf[conf] * np.std(mi) / np.sqrt(len(mi))
            else:
                all_mi[iRound] = np.mean(mi)
                all_mi_min[iRound] = np.mean(mi) - z_conf[conf] * np.std(mi) / np.sqrt(len(mi)) 
                all_mi_max[iRound] = np.mean(mi) + z_conf[conf] * np.std(mi) / np.sqrt(len(mi))
        print(all_mi)

        ax.plot(list(all_mi.keys()), list(all_mi.values()), "*-")
        ax.fill_between([1, 5, 10, 20, 30], list(all_mi_min.values()), list(all_mi_max.values()), alpha=.1)
ax.set_xlabel("Training round T", fontsize=FONTSIZE)
ax.grid(True)
if use_norm == "low":
    ax.set_ylabel("Normalized MI (%)", fontsize=FONTSIZE)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
elif use_norm == "high":
    ax.set_ylabel("Unnormalized MI (bits)", fontsize=FONTSIZE)
ax.legend(['MNIST',"CIFAR10"], fontsize=22, ncol=2)
fig.savefig(f"./results/{fig_location}/results_{version}_{use_norm}.jpg", bbox_inches='tight')
