import json
from turtle import color
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.rcParams['font.family'] = 'Times New Roman'
import numpy as np
import math

FONTSIZE = 24
# version = "avg_sgd_cifar10"
# version = "avg_sgd"
version = "may_mnist_fedprox"
# entropy = 1403 * 1000 # cifar10
entropy = 567 * 1200 # mnist
root = "./results/fedprox"

z_conf = {"80": 1.28, "90": 1.645, "95": 1.96, "98": 2.33, "99": 2.58}
conf = "95"
d = 7850
num = 1
fig_location = "final"

for tag in ["_small"]:
    if tag == "":
        user_list = [1,2,5,10,20,50]
        baseline_raw = [entropy] + [d / 2 * np.log(n / (n-1)) for n in [2,5,10,20,50]]
    else:
        user_list = [2,5,10,20,50]
        baseline_raw = [d / 2 * np.log(n / (n-1)) for n in [2,5,10,20,50]]

    for use_norm in ["high", "low"]:
        fig, ax = plt.subplots()
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.tick_params(axis='x', labelsize=FONTSIZE)
        ax.tick_params(axis='y', labelsize=FONTSIZE, colors='blue')

        # for model in ["linear", "nlinear", "cnn"]:
        # for model in ["linear", "nlinear", "fcnn"]:
        for model in ["nlinear"]:
            res = [{} for _ in range(3)]
            for num_user in user_list:
                # if model == "nlinear":
                #     with open(f"{root}/{model}/loss_{num_user}_{model}_{num_user}_avg_sgd_2_cifar10.json", "r") as json_file:
                #         data = json.load(json_file)
                # else:
                with open(f"{root}/{model}/loss_{num_user}_{model}_{num_user}_ep_1_{version}.json", "r") as json_file:
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

            ax.plot(list(res[0].keys()), list(res[0].values()), "b*-")
            ax.fill_between(user_list, list(res[1].values()), list(res[2].values()), alpha=.1)
        ax2 = ax.twinx()
        ax2.plot(list(res[0].keys()), [6.13, 5.51, 4.77, 4.68, 4.56], "r*-")
        ax2.set_ylabel('PSNR', color='r')
        ax.set_xlabel("Number of users", fontsize=FONTSIZE)
        # ax.legend(["linear, d=7890", "fcnn, d=7890", "fcnn, d=89610"], fontsize=FONTSIZE)
        # ax.legend(["Linear", "SLP", "MLP"], fontsize=FONTSIZE)
        # ax.legend(["linear, d=30730", "fcnn, d=30730", "cnn, d=82554"], fontsize=FONTSIZE)
        # ax.legend(["Linear", "SLP", "CNN"], fontsize=FONTSIZE)
        ax.grid(True)
        if use_norm == "low":
            ax.set_ylabel("Normalized MI (%)", fontsize=FONTSIZE, color='blue')
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        elif use_norm == "high":
            ax.set_ylabel("Unormalized MI (bits)", fontsize=FONTSIZE)
        ax2.set_ylabel("PSNR", fontsize=FONTSIZE)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax2.tick_params(axis='y', labelsize=FONTSIZE, colors='red')
        # fig.savefig(f"{root}/{fig_location}/results_cmp_{version}_{use_norm}_avg{tag}.eps", bbox_inches='tight')
        fig.savefig(f"{root}/{fig_location}/results_cmp_{version}_{use_norm}_avg{tag}_dlg.jpg", bbox_inches='tight')