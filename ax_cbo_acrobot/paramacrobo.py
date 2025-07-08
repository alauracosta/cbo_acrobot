
class ParamAcroBO:
    def __init__(self, reps, n_seed, n_iter, target_context, train_context, FILLING = False, th_optimum=5, vanilla_best=None, vanilla_lowest = None,
                 save_data_all=True, save_data_reps=False, plot_all=True, plot_reps=False):
        self.reps = reps
        self.n_seed = n_seed
        self.n_iter = n_iter
        self.target_context = target_context
        self.train_context = train_context
        self.FILLING = FILLING
        self.th_optimum = th_optimum
        self.vanilla_best = vanilla_best
        self.vanilla_lowest = vanilla_lowest
        self.save_data_all = save_data_all
        self.save_data_reps = save_data_reps
        self.plot_all = plot_all
        self.plot_reps = plot_reps