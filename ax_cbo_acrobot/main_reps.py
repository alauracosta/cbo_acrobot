import copy
import numpy as np
import matplotlib
import os
import warnings

from plot_all import plot_all
from paramacrobo import ParamAcroBO
from run_reps import run_reps
from utils import init_benchmark


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
        seed_vanilla= 10,
        runs_vanilla = 2, 
        save_data_all=True,
        save_data_reps=False, 
        plot_all=True,
        plot_reps=False)
    
    # initialize BO vanilla as benchmark
    init_benchmark(params)
    
    # run optimization over reps
    data = run_reps(params)

    if params.plot_all:
         plot_all(data, params)

    print(f'Optimization over {params.reps} iterations: ')