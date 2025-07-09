
import copy
import matplotlib.pyplot as plt

from acrocbo import AcroCBO
from paramacrobo import ParamAcroBO

def run_cbo(params, tuner=None):

    if params.active_context_i ==0: # we are in the first context, so we need filling
        tuner = AcroCBO(params.n_seed, params.n_iter, params.target_context, params.FILLING)
        tuner.add_params(["kp", "ki", "kd", "v"], [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (0.5, 1.5)])
        
        print("  -> Create search space")
        tuner.create_search_space()

        print("  -> Create experiment")
        tuner.build_experiment()

        if params.FILLING:
            print("  -> Create the initial dataset - FILLING")
            tuner.create_random_dataset()
        else:
            print(f"  -> Create the initial dataset - NO FILLING - Train context: {params.train_context}")
            tuner.create_random_dataset(train_context=params.train_context)
            tuner.run_optimization_loop(train_context=params.train_context)
    
    tuner.target_context = params.target_context

    print(f"  -> Start optimization loop - Target context: {tuner.target_context }")
    tuner.run_optimization_loop()

    tuner.get_values(params.save_data_reps)

    tuner.calculate_stats(params.th_optimum, params.vanilla_best, params.vanilla_lowest)

    if params.plot_reps: 
        tuner.plot_data()

    tuner.limit += tuner.n_iterations

    return tuner, [tuner.trial_indices, tuner.rewards, tuner.best_reward, tuner.av_regret, tuner.it_bad]

def run_cbo_contexts(params):
    params_aux = copy.deepcopy(params)
    tuner = None
    results = []
    for i, context in enumerate(params.target_context):
        params_aux.target_context = context
        params_aux.vanilla_best = params.vanilla_best[i]
        params_aux.vanilla_lowest = params.vanilla_lowest[i]
        params_aux.active_context_i = i
        tuner, results_context = run_cbo(params_aux, tuner)
        results.append(results_context)
    return tuner, results


def run_cbo_wrapper(args):
    return run_cbo(*args)

def run_cbo_contexts_wrapper(args):
    return run_cbo_contexts(*args)

if __name__ == "__main__":
    params = ParamAcroBO(
        reps = 2,
        n_seed = 10,
        n_iter = 20,
        target_context = [1.0, 0.6],
        train_context = 1.2, 
        FILLING = False, 
        th_optimum=5,
        seed_vanilla= 10,
        runs_vanilla = 2, 
        save_data_all=False,
        save_data_reps=False, 
        plot_all=False,
        plot_reps=True)
    
    run_cbo_contexts(params)