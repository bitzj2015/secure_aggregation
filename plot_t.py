import json
import matplotlib.pyplot as plt
import numpy as np

num_user = 50
# entropy = 1403 * 1000 # 25088 
entropy = 567 * 32 # 1924

z_conf = {"80": 1.28, "90": 1.645, "95": 1.96, "98": 2.33, "99": 2.58}
conf = "95"


version = "lin_1000_avg_t"
with open(f"./results/lin/loss_{num_user}_{version}.json", "r") as json_file:
    data = json.load(json_file)

for use_norm in ["low", "high"]:
    all_mi = {}
    all_mi_min = {}
    all_mi_max = {}
    for iRound in [1, 5, 10, 20, 30]:
        mi = []
        for k in range(10):
            res = []
            for item in data[str(k)][str(iRound)]:
                if item[1] != float("inf") and item[1] != float("nan"):
                    res.append(item[1])
            mi.append(max(res))
        if use_norm == "low":
            mi = [item / entropy * 100 for item in mi]
            all_mi[iRound] = np.mean(mi)
            all_mi_min[iRound] = np.mean(mi) - z_conf[conf] * np.std(mi) / np.sqrt(len(mi)) 
            all_mi_max[iRound] = np.mean(mi) + z_conf[conf] * np.std(mi) / np.sqrt(len(mi))
        else:
            all_mi[iRound] = np.mean(mi)
            all_mi_min[iRound] = np.mean(mi) - z_conf[conf] * np.std(mi) / np.sqrt(len(mi)) 
            all_mi_max[iRound] = np.mean(mi) + z_conf[conf] * np.std(mi) / np.sqrt(len(mi))

    fig, ax = plt.subplots()
    ax.plot(list(all_mi.keys()), list(all_mi.values()), "*-")
    ax.fill_between([1, 5, 10, 20, 30], list(all_mi_min.values()), list(all_mi_max.values()), alpha=.1)
    ax.set_xlabel("Training round T")
    if use_norm == "low":
        ax.set_ylabel("Estimated accumulative MI divided by entropy (%)")
    elif use_norm == "high":
        ax.set_ylabel("Estimated accumulative MI (bits)")
    fig.savefig(f"./results/figs_new/results_{version}_{use_norm}.jpg")

# print(all_mi)