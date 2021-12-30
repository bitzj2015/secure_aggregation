import json
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import gmean

model = "nn"
version = "avg"
use_log = False
plt.figure(1)
res = {}
for num_user in [1,2,5,10,20,50]:
    with open(f"./results/loss_{num_user}_{model}_{num_user}_{version}.json", "r") as json_file:
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
    res[num_user] = np.mean(avg_max_MI_by_round)
plt.legend(["1 user"] + [f"{num_user} users" for num_user in [2,5,10,20,50]])
plt.xlabel("Training round")
if use_log:
    plt.ylabel("MI estimated by MINE (log10)")
    plt.savefig(f"./param/results_{model}_{version}_log10.jpg")
else:
    plt.ylabel("MI estimated by MINE")
    plt.savefig(f"./param/results_{model}_{version}.jpg")

plt.figure(2)
plt.plot(list(res.keys()), list(res.values()))
plt.xlabel("Number of users")
plt.ylabel("Average MI estimated by MINE.")
plt.savefig(f"./param/results_{model}_{version}_avg.jpg")
print(res)