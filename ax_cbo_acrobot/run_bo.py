
import matplotlib.pyplot as plt

from acrobo import AcroBO

def run_bo(params):

    tuner = AcroBO(params.n_seed, params.n_iter, params.target_context)
    tuner.add_params(["kp", "ki", "kd", "v"], [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (0.5, 1.5)])
    
    print("  -> Create search space")
    tuner.create_search_space()

    print("  -> Create experiment")
    tuner.build_experiment()
    
    print("  -> Create the initial dataset")
    tuner.create_random_dataset()

    print("  -> Start optimization loop")
    tuner.run_optimization_loop()

    tuner.get_values(params.save_data_reps)

    tuner.calculate_stats(params.th_optimum, params.vanilla_best, params.vanilla_lowest)

    if params.plot_reps: 
        tuner.plot_data()

    return tuner.trial_indices, tuner.rewards, tuner.best_reward, tuner.av_regret, tuner.it_bad

def run_bo_wrapper(args):
    return run_bo(*args)
