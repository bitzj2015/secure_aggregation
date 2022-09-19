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

model = "linear"
version = "eps"
tag = "_may_mnist_fedavg"
FONTSIZE = 24
root = "./results/fedprox"

# entropy = 1403 * 1000 # 25088 
entropy = 567 * 1200 # 1924
# entropy = 678 * 176

z_conf = {"80": 1.28, "90": 1.645, "95": 1.96, "98": 2.33, "99": 2.58}
conf = "95"
d = 7850
num = 1
fig_location = "final"

# user_list = [2,5,10,20,50,1000]

# for use_norm in ["high", "low"]:
#     fig, ax = plt.subplots()
#     ax.yaxis.set_major_locator(plt.MaxNLocator(6))
#     ax.xaxis.set_major_locator(plt.MaxNLocator(6))
#     ax.tick_params(axis='x', labelsize=FONTSIZE)
#     ax.tick_params(axis='y', labelsize=FONTSIZE)
#     for bs in [1,5,10,5000]:#10000000000000]:
#         res = [{} for _ in range(3)]
#         for num_user in user_list:
#             if bs == -1:
#                 with open(f"./results/lin/loss_{num_user}_lin_{num_user}_avg.json", "r") as json_file:
#                     data = json.load(json_file)
#             else:
#                 with open(f"{root}/{model}/loss_{num_user}_{model}_{num_user}_{version}_{bs}{tag}.json", "r") as json_file:
#                     data = json.load(json_file)
#             avg_max_MI_by_round = []
#             avg_max_MI_by_round_low = []
#             avg_max_MI_by_round_high = []
#             all_max_MI_by_round = []
#             for round in range(30):
#                 max_MI_by_round = []
#                 try:
#                     for niter in range(10):
#                         try:
#                             MI = []
#                             for item in data[str(round)][str(niter)]:
#                                 if math.isnan(item[1]) == False:
#                                     if item[1] != float('inf'):
#                                         MI.append(item[1])

#                             if len(MI) == 0:
#                                 continue
#                             max_MI = np.mean(sorted(MI)[-10:])
#                             max_MI_by_round.append(max_MI)
#                         except:
#                             continue
#                 except:
#                     continue
#                 if len(max_MI_by_round) == 0:
#                     continue
#                 max_MI_by_round = sorted(max_MI_by_round)[1:-1]
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

#             res[0][num_user] = np.mean(avg_max_MI_by_round)
#             res[1][num_user] = np.mean(all_max_MI_by_round) - z_conf[conf] * np.std(all_max_MI_by_round) / np.sqrt(len(all_max_MI_by_round))
#             res[2][num_user] = np.mean(all_max_MI_by_round) + z_conf[conf] * np.std(all_max_MI_by_round) / np.sqrt(len(all_max_MI_by_round))

#         ax.plot(list(res[0].keys()), list(res[0].values()), "*-")
#         print(bs, list(res[0].values()))
#         ax.fill_between(user_list, list(res[1].values()), list(res[2].values()), alpha=.1)

#     ax.set_xlabel("Number of users", fontsize=FONTSIZE)

#     # ax.legend(["B=16", "B=32", "B=64", "B=128", "B=256"], fontsize=FONTSIZE)
#     ax.legend(["$\epsilon=1$", "$\epsilon=5$", "$\epsilon=10$", "$\epsilon=5000$"], fontsize=FONTSIZE, ncol=2, frameon=False, bbox_to_anchor=(-0.0,1,1,0.2))
#     ax.grid(True)
#     if use_norm == "low":
#         ax.set_ylabel("Normalized MI (%)", fontsize=FONTSIZE)
#         ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#     elif use_norm == "high":
#         ax.set_ylabel("Unnormalized MI (bits)", fontsize=FONTSIZE)
#     ax.set_ylim([0,0.26])
#     ax.set_xscale('log')

#     fig.savefig(f"{root}/{fig_location}/results_cmp_ep_{version}_{use_norm}_avg{tag}.jpg", bbox_inches='tight')


# fig, ax = plt.subplots()
# ax.yaxis.set_major_locator(plt.MaxNLocator(6))
# ax.xaxis.set_major_locator(plt.MaxNLocator(6))
# ax.tick_params(axis='x', labelsize=FONTSIZE)
# ax.tick_params(axis='y', labelsize=FONTSIZE)

# acc_res = {}

# for eps in [1,5,10,5000]:
#     acc_res[eps] = {}
#     for user in [2,5,10,20,50,1000]:
#         with open(f"./logs/log_dp_{user}_eps_{eps}_linear.txt", 'r') as file:
#             acc = 0
#             cnt = 0
#             for line in file.readlines():
#                 cnt += 1
#                 if cnt <= 2:
#                     continue
#                 acc = float(line.replace('\n', '').split(' ')[-1])
#             acc_res[eps][user] = acc
    
#     ax.plot(list(acc_res[eps].keys()), [item * 100 for item in list(acc_res[eps].values())], "*-")
#     print(acc_res)

# ax.set_xlabel("Number of users", fontsize=FONTSIZE)
# ax.set_ylabel("Model accuracy (%)", fontsize=FONTSIZE)
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.f'))
# ax.legend(["$\epsilon=1$", "$\epsilon=5$", "$\epsilon=10$", "$\epsilon=5000$"], fontsize=FONTSIZE, ncol=2, frameon=False, bbox_to_anchor=(-0.1,1,1,0.2))
# ax.grid(True)
# ax.set_xscale('log')
# fig.savefig(f"{root}/{fig_location}/results_cdp_acc.jpg", bbox_inches='tight')



fig, ax = plt.subplots()
ax.yaxis.set_major_locator(plt.MaxNLocator(6))
ax.xaxis.set_major_locator(plt.MaxNLocator(6))
ax.tick_params(axis='x', labelsize=FONTSIZE)
ax.tick_params(axis='y', labelsize=FONTSIZE)

prox_mi_mnist = {1: 0.49687000882408333, 5: 0.9467714322026793, 10: 1.1540994382899765}
prox_mi_cifar10 = {1: 0.21882725623872398, 5: 0.27484385788823057, 10: 0.43548007826421065}
avg_mi_mnist = {1: 0.023777724605388743, 5: 0.35091843647090637, 10: 0.8763660643957142}
avg_mi_cifar10 = {1: 0.6524506219930506, 5: 0.7862063087468817, 10: 0.8527030386894158}
sgd_mi_mnist = {1: 0.023777724605388743, 5: 0.35091843647090637, 10: 0.8763660643957142, 50: 2.41}
sgd_mi_cifar10 = {1: 0.6524506219930506, 5: 0.7862063087468817, 10: 0.8527030386894158, 50: 1.09}

prox_acc_mnist = {1: 87.36, 5: 90.68, 10: 91.35}
prox_acc_cifar10 = {1: 36.68, 5: 38.32, 10: 36.73}
avg_acc_mnist = {1: 87.34, 5: 90.74, 10: 90.84}
avg_acc_cifar10 = {1: 36.73, 5: 38.32, 10: 36.60}
sgd_acc_mnist = {1: 15.47, 5: 77.56, 10: 83.48, 50: 89.12}
sgd_acc_cifar10 = {1: 21.82, 5: 30.23, 10: 32.16, 50: 37.38}

# ax.plot(list(sgd_acc_mnist.values()), list(sgd_mi_mnist.values()), "*-")
# ax.plot(list(avg_acc_mnist.values()), list(avg_mi_mnist.values()), "*-")
# ax.plot(list(prox_acc_mnist.values()), list(prox_mi_mnist.values()), "*-")

ax.plot(list(sgd_acc_cifar10.values()), list(sgd_mi_cifar10.values()), "*-")
ax.plot(list(avg_acc_cifar10.values()), list(avg_mi_cifar10.values()), "*-")
ax.plot(list(prox_acc_cifar10.values()), list(prox_mi_cifar10.values()), "*-")

ax.set_xlabel("Model accuracy (%)", fontsize=FONTSIZE)
ax.set_ylabel("Normalized MI (%)", fontsize=FONTSIZE)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.legend(["FedSGD", "FedAvg", "FedProx"], fontsize=FONTSIZE)
ax.grid(True)
fig.savefig(f"{root}/{fig_location}/results_accum_acc_cifar.jpg", bbox_inches='tight')
