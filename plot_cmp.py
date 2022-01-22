import json
from locale import normalize
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import gmean

model = "nn"
version = "avg_sgd"

for tag in ["", "_small"]:
    if tag == "":
        user_list = [1,2,5,10,20,50]
    else:
        user_list = [2,5,10,20,50]

    for use_norm in ["low", "high"]:
        fig, ax = plt.subplots()

        for model in ["lin", "nlin", "nn"]:
            res = [{} for _ in range(3)]
            for num_user in user_list:
                with open(f"./results/{model}/loss_{num_user}_{model}_{num_user}_{version}.json", "r") as json_file:
                    data = json.load(json_file)
                
                avg_max_MI_by_round = []
                avg_max_MI_by_round_low = []
                avg_max_MI_by_round_high = []
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
                            max_MI_by_round.append(max_MI)
                    except:
                        continue
                    if use_norm == "low":
                        avg_max_MI_by_round.append(np.mean(max_MI_by_round) / 1924 * 100)
                        avg_max_MI_by_round_low.append(sorted(max_MI_by_round)[0] / 1924 * 100)
                        avg_max_MI_by_round_high.append(sorted(max_MI_by_round)[-1] / 1924 * 100)
                    elif use_norm == "high":
                        avg_max_MI_by_round.append(np.mean(max_MI_by_round) / 1200)
                        avg_max_MI_by_round_low.append(sorted(max_MI_by_round)[0] / 1200)
                        avg_max_MI_by_round_high.append(sorted(max_MI_by_round)[-1] / 1200)

                res[0][num_user] = np.mean(avg_max_MI_by_round)
                res[1][num_user] = np.mean(avg_max_MI_by_round_low)
                res[2][num_user] = np.mean(avg_max_MI_by_round_high)

            ax.plot(list(res[0].keys()), list(res[0].values()))
            ax.fill_between(user_list, list(res[1].values()), list(res[2].values()), alpha=.1)
        ax.set_xlabel("Number of users")
        ax.legend(["linear, d=7890", "non-linear, d=7890", "non-linear, d=89610"])
        if use_norm == "low":
            ax.set_ylabel("Estimated MI divided by entropy (%)")
        elif use_norm == "high":
            ax.set_ylabel("Estimated MI (bits/image)")
        fig.savefig(f"./results/figs/results_cmp_{version}_{use_norm}_avg{tag}.jpg")
