import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import statistics as stat

from paramacrobo import ParamAcroBO
from utils import load_data

def plot_all(data, params):

    trial_indices = data.reps_dic['trial_index']
    rewards = data.reps_dic['mean']
    rewards_min  = data.reps_dic['reward_min']
    rewards_max  = data.reps_dic['reward_max']
    reward_std_dev = data.reps_dic['std_dev'] 
    limit = data.limit

    # plotting the evolution over iterations
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(trial_indices[limit:], rewards[limit:], marker='o',  color='blue', label = 'mean - optimization')
    #ax.plot(trial_indices[:limit], rewards[:limit], marker='o', label = 'filling space')
    ax.fill_between(trial_indices[limit:], rewards_min[limit:], rewards_max[limit:], color='blue', alpha=0.4, label='spread - optimization')
    ax.fill_between(trial_indices[:limit], rewards_min[:limit], rewards_max[:limit], color='orange', alpha=0.4, label='spread - random sampling')
    lower_bound = np.array(rewards[limit:]) - np.array(reward_std_dev[limit:])
    upper_bound = np.array(rewards[limit:]) + np.array(reward_std_dev[limit:])
    ax.plot(trial_indices[limit:], lower_bound, linestyle=':',  color='blue', alpha=0.8, label = 'std dev interval - optimization')
    ax.plot(trial_indices[limit:], upper_bound, linestyle=':',  color='blue', alpha=0.8)
    ax.axvline(trial_indices[limit], color='black', linestyle='dotted')
    plt.xlabel("Trial Index")
    plt.ylabel("Mean Reward")
    plt.legend()
    plt.title(f"{data.title}\nMean: {stat.mean(rewards):.2f}, Avg Best: {data.mean_av_dic['best']:.2f}, Avg Regret: {data.mean_av_dic['av_regret']:.2f}, Avg It bad:{data.mean_av_dic['it_bad']:.2f} ")

    save_folder = "results\\"
    os.makedirs(save_folder, exist_ok=True)  
    fig_save_path = data.base_path + "_PLOT.png"
    plt.savefig(fig_save_path)
    print(f"Figure saved in {save_folder}")

    plt.show()

    # histogram of rewards distribution
    rewards_total = []
    for i in range(params.reps):
        rewards_total.extend(data.reps_dic[f'rep_{i}'][limit:])

    plt.figure()
    plt.hist(rewards_total, bins=np.arange(min(rewards_total), max(rewards_total) + 5, 5), edgecolor='black')
    plt.xlabel('Mean Value')
    plt.ylabel('Frequency')
    plt.title(f'{data.title}\n Histogram of Total Rewards')

    save_folder = "results\\"
    os.makedirs(save_folder, exist_ok=True)  
    fig_save_path = data.base_path + "_HIST.png"
    plt.savefig(fig_save_path)
    print(f"Figure saved in {save_folder}")

    plt.show()



if __name__ == "__main__":
    os.chdir(r"Z:\\GitHub\\acrobot_tutorial\\ax_cbo_acrobot")

    params = ParamAcroBO(
        reps = 10,
        n_seed = 10,
        n_iter = 20,
        target_context = 1.0,
        train_context = 1.2, 
        FILLING = False, 
        th_optimum=5,
        vanilla_best=None,
        vanilla_lowest = None,
        save_data_all=True,
        save_data_reps=False, 
        plot_all=True,
        plot_reps=False)
    
    data = load_data(params)

    plot_all(data, params)