import json
import matplotlib.pyplot as plt
import numpy as np

num_user = 20

version = "lin_20_avg_t"
with open(f"./results/loss_{num_user}_{version}.json", "r") as json_file:
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
            all_mi[iRound] = np.mean(mi) / 1924 * 100
            all_mi_min[iRound] = np.min(mi) / 1924 * 100
            all_mi_max[iRound] = np.max(mi) / 1924 * 100
        else:
            all_mi[iRound] = np.mean(mi) / 1200
            all_mi_min[iRound] = np.min(mi) / 1200
            all_mi_max[iRound] = np.max(mi) / 1200

    fig, ax = plt.subplots()
    ax.plot(list(all_mi.keys()), list(all_mi.values()), "*-")
    ax.fill_between([1, 5, 10, 20, 30], list(all_mi_min.values()), list(all_mi_max.values()), alpha=.1)
    ax.set_xlabel("Training round T")
    if use_norm == "low":
        ax.set_ylabel("Estimated accumulative MI divided by entropy (%)")
    elif use_norm == "high":
        ax.set_ylabel("Estimated accumulative MI (bits/image)")
    fig.savefig(f"./results/figs/results_{version}_{use_norm}.jpg")

