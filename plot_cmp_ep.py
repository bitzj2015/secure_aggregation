import json
from locale import normalize
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import gmean

model = "nn"
version = "avg"
# entropy = 1403 * 1000 # 25088 
entropy = 567 * 32 # 1924

z_conf = {"80": 1.28, "90": 1.645, "95": 1.96, "98": 2.33, "99": 2.58}
conf = "95"
d = 7850
num = 1

for tag in ["", "_small"]:
    if tag == "":
        user_list = [1,2,5,10,20,50]
    else:
        user_list = [2,5,10,20,50]

    for use_norm in ["low", "high"]:
        fig, ax = plt.subplots()

        for ep in [1,2,5,10]:
            res = [{} for _ in range(3)]
            for num_user in user_list:
                with open(f"./results/{model}/loss_{num_user}_nn_{num_user}_{version}_{ep}.json", "r") as json_file:
                    data = json.load(json_file)
                
                avg_max_MI_by_round = []
                avg_max_MI_by_round_low = []
                avg_max_MI_by_round_high = []
                all_max_MI_by_round = []
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
                        avg_max_MI_by_round.append(np.mean(max_MI_by_round) / entropy * 100)
                        # avg_max_MI_by_round_low.append(sorted(max_MI_by_round)[0] / entropy * 100)
                        # avg_max_MI_by_round_high.append(sorted(max_MI_by_round)[-1] / entropy * 100)
                        max_MI_by_round = [item / entropy * 100 for item in max_MI_by_round]

                    elif use_norm == "high":
                        avg_max_MI_by_round.append(np.mean(max_MI_by_round) / num)
                        # avg_max_MI_by_round_low.append(sorted(max_MI_by_round)[0] / num)
                        # avg_max_MI_by_round_high.append(sorted(max_MI_by_round)[-1] / num)
                        max_MI_by_round = [item / num for item in max_MI_by_round]

                    avg_max_MI_by_round_low.append(np.mean(max_MI_by_round) - z_conf[conf] * np.std(max_MI_by_round) / np.sqrt(len(max_MI_by_round)))
                    avg_max_MI_by_round_high.append(np.mean(max_MI_by_round) + z_conf[conf] * np.std(max_MI_by_round) / np.sqrt(len(max_MI_by_round)))
                    all_max_MI_by_round += max_MI_by_round

                res[0][num_user] = np.mean(avg_max_MI_by_round)
                res[1][num_user] = np.mean(all_max_MI_by_round) - z_conf[conf] * np.std(all_max_MI_by_round) / np.sqrt(len(all_max_MI_by_round))
                res[2][num_user] = np.mean(all_max_MI_by_round) + z_conf[conf] * np.std(all_max_MI_by_round) / np.sqrt(len(all_max_MI_by_round))

            ax.plot(list(res[0].keys()), list(res[0].values()), "*-")
            ax.fill_between(user_list, list(res[1].values()), list(res[2].values()), alpha=.1)
        ax.set_xlabel("Number of users")
        ax.legend(["E=1", "E=2", "E=5", "E=10"])
        if use_norm == "low":
            ax.set_ylabel("Estimated MI divided by entropy (%)")
        elif use_norm == "high":
            ax.set_ylabel("Estimated MI (bits)")
        fig.savefig(f"./results/figs_new/results_cmp_ep_{version}_{use_norm}_avg{tag}.jpg")
