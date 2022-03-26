import json
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import gmean

model = "nn"
version = "bs"
use_log = False
num_user = 20
plt.figure(1)
res = {}
for num_ep in [16,32,64,128,256]:
    with open(f"./results/loss_{num_user}_{model}_{num_user}_{version}_{num_ep}.json", "r") as json_file:
        data = json.load(json_file)
    
    avg_max_MI_by_round = []
    for round in range(30):
        max_MI_by_round = []
        try:
            for niter in range(10):
                MI = []
                for item in data[str(round)][str(niter)]:
                    if item[1] != float('inf'):
                        if math.isnan(item[1]) == False:
                            MI.append(item[1])
                max_MI = sorted(MI)[-1]
                if use_log:
                    max_MI_by_round.append(np.log10(max_MI))
                else:
                    max_MI_by_round.append(max_MI)
        except:
            continue
        avg_max_MI_by_round.append(np.mean(max_MI_by_round))
    plt.plot(avg_max_MI_by_round)
    res[num_ep] = np.mean(avg_max_MI_by_round)
plt.legend([f"{num_ep} bs" for num_ep in [16,32,64,128,256]])
plt.xlabel("Batch size")
if use_log:
    plt.ylabel("MI estimated by MINE (log10)")
    plt.savefig(f"./param/results_{num_user}_{model}_{version}bs_log10.jpg")
else:
    plt.ylabel("MI estimated by MINE")
    plt.savefig(f"./param/results_{num_user}_{model}_{version}bs.jpg")

plt.figure(2)
plt.plot(list(res.keys()), list(res.values()))
plt.xlabel("Batch size in FL")
plt.ylabel("Average MI estimated by MINE.")
plt.savefig(f"./param/results_{num_user}_{model}_{version}bs_avg.jpg")
print(res)