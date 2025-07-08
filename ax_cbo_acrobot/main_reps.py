import copy
import numpy as np
import matplotlib
import os
import warnings

from plot_all import plot_all
from paramacrobo import ParamAcroBO
from run_reps import run_reps
from utils import load_data


# Plotting imports, initialization and warnings
matplotlib.use("TkAgg")
warnings.filterwarnings("ignore", category=UserWarning, module="gym") 
os.chdir(r"Z:\\GitHub\\acrobot_tutorial\\ax_cbo_acrobot")

seed_vanilla = 10
runs_vanilla = 2

if __name__ == "__main__":
    params = ParamAcroBO(
        reps = 2,
        n_seed = 10,
        n_iter = 20,
        target_context = 1.0,
        train_context = False, 
        FILLING = False, 
        th_optimum=5,
        vanilla_best=None,
        vanilla_lowest = None,
        save_data_all=True,
        save_data_reps=False, 
        plot_all=True,
        plot_reps=False)
    
    # initialize BO vanilla as benchmark
    if params.n_seed < 10 or params.train_context:
        print('Running Contextual BO') if params.train_context else print('Running Low Sampling BO')
        params_vanilla = copy.deepcopy(params)
        params_vanilla.n_seed = seed_vanilla 
        params_vanilla.train_context = False
        try:
            data_vanilla = load_data(params_vanilla)
        except Exception as e:
            print("Run Vanilla Mehtod to obtain benchmark")
            params_vanilla.reps = runs_vanilla
            data_vanilla = run_reps(params_vanilla)
        params.vanilla_best = (max(data_vanilla.av_dic['best']))
        params.vanilla_lowest = (np.min(data_vanilla.rewards_matrix[data_vanilla.limit:,]))
    
    # run optimization over reps
    data = run_reps(params)

    if params.plot_all:
         plot_all(data, params)

    print(f'Optimization over {params.reps} iterations: ')