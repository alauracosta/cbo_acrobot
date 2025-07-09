
import copy
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import statistics as stat

class PostProcessedDataAcro:
    def __init__(self,reps_dic, av_dic, mean_av_dic, rewards_matrix, limit, title, base_path):
        self.reps_dic = reps_dic
        self.av_dic = av_dic
        self.mean_av_dic = mean_av_dic
        self.rewards_matrix = rewards_matrix
        self.limit = limit
        self.title = title
        self.base_path = base_path
                 

def handle_working_directory():
    """Ask user if they want to stay in current working directory or change it."""
    print("Now working in:", os.getcwd())
    choice = input("Do you want to stay in this path? (y/n): ").strip().lower()

    if choice == 'n':
        new_path = input("Enter new path: ").strip()
        if os.path.isdir(new_path):
            os.chdir(new_path)
            print("Changed working directory to:", os.getcwd())
        else:
            print("Invalid path. Staying in the current directory:", os.getcwd())
    else:
        print("Staying in current directory:", os.getcwd())

def extend_or_default(value, n):
    if value is None:
        return [None] * n
    elif len(value) < n:
        return value + [None] * (n - len(value))
    else:
        return value
    

def init_benchmark(params):
    from run_reps import run_reps
    if params.n_seed < 10 or params.train_context:
        print('Running Contextual BO') if params.train_context else print('Running Low Sampling BO')
        params_vanilla = copy.deepcopy(params)
        params_vanilla.n_seed = params.seed_vanilla 
        params_vanilla.train_context = False
        try:
            data_vanilla = load_data(params_vanilla)
        except Exception as e:
            print("Run Vanilla Mehtod to obtain benchmark")
            params_vanilla.reps = params.runs_vanilla
            data_vanilla = run_reps(params_vanilla)
        params.vanilla_best = (max(data_vanilla.av_dic['best']))
        params.vanilla_lowest = (np.min(data_vanilla.rewards_matrix[data_vanilla.limit:,]))

def save_data_reps(params, df_reps, df_averaged):
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


        
def load_data(params):

    if params.train_context: 
        method = "acrocbo"
        if params.FILLING:
            limit = params.n_seed
        else:
            limit = params.n_seed + params.n_iter
        filling_str = "filling" if params.FILLING else f"train{params.train_context}" 
        base_path = f"results\{method}_r{params.reps}_target{params.target_context}_{filling_str}_s{params.n_seed}it{params.n_iter}"
        paths = [f"{base_path}_REPS.csv", f"{base_path}_AV.csv"]
        title = f"CBO for {params.target_context} with {filling_str} - over {params.reps} iterations"
    else:
        method = "acrobo"
        limit = params.n_seed
        base_path = f"results\{method}_r{params.reps}_target{params.target_context}_s{params.n_seed}it{params.n_iter}"
        paths = [f"{base_path}_REPS.csv", f"{base_path}_AV.csv"]
        title = f"BO for {params.target_context}- over {params.reps} iterations"

    total_data = []
    for path in paths:
        with open(path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            data = list(reader)
        total_data.append(data)
    

    reps_data = total_data[0]
    reps_dic = {
        'trial_index': [],
        'mean': [],
        'reward_max': [],
        'reward_min': [],
    }

    for i in range(params.reps):
        reps_dic[f'rep_{i}'] = []

    for row in reps_data:
        reps_dic['trial_index'].append(int(row['trial_index']))
        reps_dic['mean'].append(float(row['mean']))
        reps_dic['reward_max'].append(float(row['reward_max']))
        reps_dic['reward_min'].append(float(row['reward_min']))
        for i in range(params.reps):
            reps_dic[f'rep_{i}'].append(float(row[f'rep_{i}']))

    rewards_matrix = np.array([reps_dic[f'rep_{i}'] for i in range(params.reps)]).T #shape (trials, reps)
    std_dev = np.std(rewards_matrix, axis=1)
    reps_dic['std_dev'] = std_dev.tolist()

    av_data = total_data[1]
    av_dic = {'iteration': [], 'best': [], 'av_regret': [], 'it_bad': []}

    for row in av_data:
        av_dic['iteration'].append(int(row['iteration']))
        av_dic['best'].append(float(row['best']))
        av_dic['av_regret'].append(float(row['av_regret']))
        av_dic['it_bad'].append(float(row['it_bad']))
    
    mean_av_dic = {'best': stat.mean(av_dic['best']), 'av_regret': stat.mean(av_dic['av_regret']) , 'it_bad':  stat.mean(av_dic['it_bad'])}

    data = PostProcessedDataAcro(reps_dic, av_dic, mean_av_dic, rewards_matrix, limit, title, base_path)
    
    return data


