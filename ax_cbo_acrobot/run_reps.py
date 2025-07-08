import os
import numpy as np
import pandas as pd
from multiprocessing import Pool

from paramacrobo import ParamAcroBO
from run_bo_ax import run_bo_ax_wrapper
from run_cbo_ax import run_cbo_ax_wrapper
from utils import handle_working_directory, load_data


def run_reps(params):
    """Run Bayesian optimization with multiprocessing across repetitions."""
    handle_working_directory()

    # Prepare arguments for each repetition
    bo_args = [
        (params,) 
        for _ in range(params.reps)
    ]

    # Parallel execution
    if params.train_context:
        with Pool(processes=os.cpu_count()) as pool:
            results = pool.map(run_cbo_ax_wrapper, bo_args)
    else:
        with Pool(processes=os.cpu_count()) as pool:
            results = pool.map(run_bo_ax_wrapper, bo_args)
        

    # Collect results
    rewards_list, best_list, av_regret_list, it_bad_list = [], [], [], []

    for trial_indices, rewards, best, av_regret, it_bad in results:
        rewards_list.append(rewards)
        best_list.append(best)
        av_regret_list.append(av_regret)
        it_bad_list.append(it_bad)

    rewards_matrix = np.column_stack(rewards_list)

    df_reps = pd.DataFrame(rewards_matrix, columns=[f"rep_{i}" for i in range(params.reps)])
    df_reps.insert(0, "trial_index", trial_indices)
    df_reps["mean"] = np.mean(rewards_matrix, axis=1)
    df_reps["reward_max"] = np.max(rewards_matrix, axis=1)
    df_reps["reward_min"] = np.min(rewards_matrix, axis=1)

    df_averaged = pd.DataFrame({
        "iteration": range(params.reps),
        "best": best_list,
        "av_regret": av_regret_list,
        "it_bad": it_bad_list
    })

    if params.save_data_all:
        save_folder = "results"
        os.makedirs(save_folder, exist_ok=True)
        if params.train_context:
            filling_str = "filling" if params.FILLING else f"train{params.train_context}" 
            base_filename = f"results/acrocbo_r{params.reps}_target{params.target_context}_{filling_str}_s{params.n_seed}it{params.n_iter}"
        else:
            base_filename = f"results/acrobo_r{params.reps}_target{params.target_context}_s{params.n_seed}it{params.n_iter}"
        df_reps.to_csv(f"{base_filename}_REPS.csv", index=False)
        df_averaged.to_csv(f"{base_filename}_AV.csv", index=False)
        print("Files saved successfully.")
       
    postdata = load_data(params)

    return postdata
    







    