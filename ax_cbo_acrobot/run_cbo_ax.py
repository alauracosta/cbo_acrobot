
import matplotlib.pyplot as plt

from acrocbo import AcroCBO

def run_cbo_ax(params):

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
        print(f"  -> Create the initial dataset - NO FILLING - Train context: {params.target_context}")
        tuner.create_random_dataset(train_context=params.target_context)
        tuner.run_optimization_loop(train_context=params.target_context)

    print("  -> Start optimization loop")
    tuner.run_optimization_loop()

    tuner.get_values(params.save_data_reps)

    tuner.calculate_stats(params.th_optimum, params.vanilla_best, params.vanilla_lowest)

    if params.plot_reps: 
        tuner.plot_data()

    return tuner.trial_indices, tuner.rewards, tuner.best_reward, tuner.av_regret, tuner.it_bad


def run_cbo_ax_wrapper(args):
    return run_cbo_ax(*args)
