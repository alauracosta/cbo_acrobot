from utils import extend_or_default

class ParamAcroBO:
    def __init__(self, reps, n_seed, n_iter, target_context, train_context, FILLING = False, th_optimum=5, vanilla_best=None, vanilla_lowest = None, seed_vanilla= 10,
                 runs_vanilla = 2, save_data_all=True, save_data_reps=False, plot_all=True, plot_reps=False):
        self.reps = reps
        self.n_seed = n_seed
        self.n_iter = n_iter
        self.target_context = target_context
        self.active_context_i = 0
        self.train_context = train_context
        self.FILLING = FILLING
        self.th_optimum = th_optimum
        self.seed_vanilla = seed_vanilla
        self.runs_vanilla = runs_vanilla
        self.save_data_all = save_data_all
        self.save_data_reps = save_data_reps
        self.plot_all = plot_all
        self.plot_reps = plot_reps

        if isinstance(target_context, list):
            n_contexts = len(target_context)

            self.vanilla_best = extend_or_default(vanilla_best, n_contexts)
            self.vanilla_lowest = extend_or_default(vanilla_lowest, n_contexts)
        else:
            self.vanilla_best = vanilla_best
            self.vanilla_lowest = vanilla_lowest