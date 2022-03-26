import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.rcParams['font.family'] = 'Times New Roman'
import numpy as np
import math
from scipy.stats import gmean

model = "cnn"
version = "bs"
tag = "_new_cifar10"
FONTSIZE = 24

entropy = 1403 * 1000 # 25088 
# entropy = 567 * 32 # 1924

z_conf = {"80": 1.28, "90": 1.645, "95": 1.96, "98": 2.33, "99": 2.58}
conf = "95"
d = 7850
num = 1
fig_location = "final"

user_list = [2,5,10,20,50]

for use_norm in ["high", "low"]:
    fig, ax = plt.subplots()
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.tick_params(axis='x', labelsize=FONTSIZE)
    ax.tick_params(axis='y', labelsize=FONTSIZE)
    for bs in [16, 32, 64, 128, 256]:
        entropy = 1403 * 1000
        res = [{} for _ in range(3)]
        for num_user in user_list:
            # if bs == 32:
            #     with open(f"./results/{model}/loss_{num_user}_{model}_{num_user}_avg{tag}.json", "r") as json_file:
            #         data = json.load(json_file)
            # else:
            with open(f"./results/{model}/loss_{num_user}_{model}_{num_user}_{version}_{bs}{tag}.json", "r") as json_file:
                data = json.load(json_file)
            
            avg_max_MI_by_round = []
            avg_max_MI_by_round_low = []
            avg_max_MI_by_round_high = []
            all_max_MI_by_round = []
            for round in range(30):
                max_MI_by_round = []
                try:
                    for niter in range(5):
                        MI = []
                        for item in data[str(round)][str(niter)]:
                            if item[1] != float('inf'):
                                if math.isnan(item[1]) == False:
                                    MI.append(item[1])
                        max_MI = np.mean(sorted(MI)[-10:])
                        max_MI_by_round.append(max_MI)
                except:
                    continue
                # print(len(max_MI_by_round))
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
    ax.set_xlabel("Number of users", fontsize=FONTSIZE)
    ax.legend(["B=16", "B=32", "B=64", "B=128", "B=256"], fontsize=FONTSIZE)
    ax.grid(True)
    if use_norm == "low":
        ax.set_ylabel("Normalized MI (%)", fontsize=FONTSIZE)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    elif use_norm == "high":
        ax.set_ylabel("Unnormalized MI (bits)", fontsize=FONTSIZE)
    fig.savefig(f"./results/{fig_location}/results_cmp_bs_{version}_{use_norm}_avg{tag}.jpg", bbox_inches='tight')

# for use_norm in ["low", "high"]:
#     fig, ax = plt.subplots()
#     ax.yaxis.set_major_locator(plt.MaxNLocator(6))
#     ax.xaxis.set_major_locator(plt.MaxNLocator(6))
#     ax.tick_params(axis='x', labelsize=FONTSIZE)
#     ax.tick_params(axis='y', labelsize=FONTSIZE)

#     for num_user in user_list:
#         res = [{} for _ in range(3)]
#         for bs in [16, 32, 64, 128, 256]:
#             # if bs == 32:
#             #     with open(f"./results/{model}/loss_{num_user}_{model}_{num_user}_avg_cifar10.json", "r") as json_file:
#             #         data = json.load(json_file)
#             # else:
#             with open(f"./results/{model}/loss_{num_user}_{model}_{num_user}_{version}_{bs}{tag}.json", "r") as json_file:
#                 data = json.load(json_file)
            
#             avg_max_MI_by_round = []
#             avg_max_MI_by_round_low = []
#             avg_max_MI_by_round_high = []
#             all_max_MI_by_round = []
#             for round in range(30):
#                 max_MI_by_round = []
#                 try:
#                     for niter in range(10):
#                         MI = []
#                         for item in data[str(round)][str(niter)]:
#                             if item[1] != float('inf'):
#                                 if math.isnan(item[1]) == False:
#                                     MI.append(item[1])
#                         max_MI = sorted(MI)[-1]
#                         max_MI_by_round.append(max_MI)
#                 except:
#                     continue

#                 if use_norm == "low":
#                     avg_max_MI_by_round.append(np.mean(max_MI_by_round) / entropy * 100)
#                     # avg_max_MI_by_round_low.append(sorted(max_MI_by_round)[0] / entropy * 100)
#                     # avg_max_MI_by_round_high.append(sorted(max_MI_by_round)[-1] / entropy * 100)
#                     max_MI_by_round = [item / entropy * 100 for item in max_MI_by_round]

#                 elif use_norm == "high":
#                     avg_max_MI_by_round.append(np.mean(max_MI_by_round) / num)
#                     # avg_max_MI_by_round_low.append(sorted(max_MI_by_round)[0] / num)
#                     # avg_max_MI_by_round_high.append(sorted(max_MI_by_round)[-1] / num)
#                     max_MI_by_round = [item / num for item in max_MI_by_round]

#                 avg_max_MI_by_round_low.append(np.mean(max_MI_by_round) - z_conf[conf] * np.std(max_MI_by_round) / np.sqrt(len(max_MI_by_round)))
#                 avg_max_MI_by_round_high.append(np.mean(max_MI_by_round) + z_conf[conf] * np.std(max_MI_by_round) / np.sqrt(len(max_MI_by_round)))
#                 all_max_MI_by_round += max_MI_by_round

#             print(len(avg_max_MI_by_round), np.std(avg_max_MI_by_round), )

#             res[0][bs] = np.mean(all_max_MI_by_round)
#             res[1][bs] = np.mean(all_max_MI_by_round) - z_conf[conf] * np.std(all_max_MI_by_round) / np.sqrt(len(all_max_MI_by_round))
#             res[2][bs] = np.mean(all_max_MI_by_round) + z_conf[conf] * np.std(all_max_MI_by_round) / np.sqrt(len(all_max_MI_by_round))

#         ax.plot(list(res[0].keys()), list(res[0].values()), "*-")
#         ax.fill_between([16, 32, 64, 128, 256], list(res[1].values()), list(res[2].values()), alpha=.1)
#     ax.set_xlabel("Batch size in FL", fontsize=FONTSIZE)
#     ax.legend([f"{num_user} users" for num_user in [2,5,10,20,50]], fontsize=FONTSIZE, ncol = 3)
#     if use_norm == "low":
#         ax.set_ylabel("Normalized MI (%)", fontsize=FONTSIZE)
#         ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#     elif use_norm == "high":
#         ax.set_ylabel("Unnormalized MI (%)", fontsize=FONTSIZE)
#     fig.savefig(f"./results/{fig_location}/results_cmp_n_bs_{version}_{use_norm}_avg{tag}.jpg", bbox_inches='tight')
