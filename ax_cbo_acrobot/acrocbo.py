import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics as stat
import torch
import os

from ax import (
    RangeParameter,
    ParameterType,
    SearchSpace,
    OptimizationConfig,
    Objective,
    Experiment,
    Models,
    Data,

)

from ax.core.runner import Runner
from ax import ParameterType, RangeParameter, SearchSpace, Experiment, Objective, OptimizationConfig
from ax.core.metric import Metric
from ax.core.observation import ObservationFeatures
from ax import Metric
from ax.modelbridge.modelbridge_utils import observed_hypervolume
from ax.models.torch.botorch_modular.surrogate import Surrogate
from botorch.models.gp_regression import SingleTaskGP, FixedNoiseGP
from botorch.acquisition.monte_carlo import qExpectedImprovement
from gymnasium.envs.classic_control import AcrobotEnv
from gpytorch.mlls.exact_marginal_log_likelihood import (
    ExactMarginalLogLikelihood,
)

from myacrobotenv import pid_to_reward

class AcroCBO:
    def __init__(self, n_seed, n_iter, target = 1.0, FILLING = None):
        self.parameters = {}
        self.n_seed = n_seed
        self.n_iterations = n_iter
        self.target_context = target
        self.experiment = None
        self.search_space = None
        self.FILLING = FILLING

        if FILLING:
            self.limit = n_seed
        else:
            self.limit = n_seed + n_iter
        

    def add_params(self, names, ranges):
        """Add parameters to the search space."""
        for i, name in enumerate(names):
            self.parameters[name] = ranges[i]

    def create_search_space(self):
        """Create the search space using the parameters """

        # Create the parameters
        x = [
            RangeParameter(
                name=name,
                lower=lb,
                upper=ub,
                parameter_type=ParameterType.FLOAT,
            )
            for name, (lb, ub) in self.parameters.items()
        ]

        # Define the search space with the parameters and constraints
        self.search_space = SearchSpace(
            parameters=[elt for elt in x],
        )

    def build_experiment(self):
        """Create the experiment using the search space and the parameters"""

        self.experiment = Experiment(
            name="contextual_bo_acrobot",
            search_space=self.search_space,
            optimization_config=OptimizationConfig(
                objective=Objective(metric=Metric(name="reward"), minimize=False),
            ),
            runner=DummyRunner()
        )

    def create_random_dataset(self, train_context = None):
        """Create the original dataset from the random sampling."""
        self.train_context = train_context

        sobol = Models.SOBOL(search_space=self.experiment.search_space)
        sobol_trials = []

        for i in range(self.n_seed):
            if train_context == None:
                new_point = sobol.gen(1)
            else: 
                new_point = sobol.gen(1, fixed_features=ObservationFeatures(parameters={'v': train_context}))
            trial = self.experiment.new_trial(generator_run=new_point) # Add generator run to a trial to make it part of the experiment and evaluate arm(s) in it
            trial.runner = self.experiment.runner 
            trial.mark_running()  # Start trial run to evaluate arm(s) in the trial
            arm = trial.arm
            reward_data = self.run_experiment(arm.parameters, trial.index, arm.name) # Get reward from experiment
            self.experiment.attach_data(reward_data)
            trial.mark_completed() # Mark as completed, to allow fetching of the data
            sobol_trials.append(trial)

        self.data = self.experiment.fetch_data()

    def run_optimization_loop(self, train_context = None):
        """Run the optimization loop of n_iterations."""

        if self.data.df.empty:
            raise ValueError("No data available to run optimization. Make sure create_random_dataset() has been run.")

        for i in range(self.n_iterations):
            #print(f"    - Iteration {i + 1}")

            model = Models.BOTORCH_MODULAR( # Reinitialize GP+EI model at each step with updated data.
                experiment=self.experiment,
                data=self.data,
                surrogate=Surrogate(
                    botorch_model_class=FixedNoiseGP,
                    mll_class=ExactMarginalLogLikelihood,
                ),
                botorch_acqf_class=qExpectedImprovement,
            )  

            # fill data with optimal values for training contexts (to improve perf) or run optimization for target context
            if train_context == None:
                generator_run = model.gen(1, fixed_features=ObservationFeatures(parameters={'v': self.target_context}))
            else: 
                generator_run = model.gen(1, fixed_features=ObservationFeatures(parameters={'v': train_context}))

            trial = self.experiment.new_trial(generator_run=generator_run)
            trial.runner = self.experiment.runner
            trial.mark_running()
            arm= trial.arm
            reward_data = self.run_experiment(arm.parameters, trial.index, arm.name
                                              )
            #print("Reward:\n", reward_data.df['mean'])
            self.experiment.attach_data(reward_data) # Attach data as cache to experiment, to then be fetched
            trial.mark_completed()
            self.data = self.experiment.fetch_data() # Update data


    def run_experiment(self, x, trial_index, arm_name):
        """Run the experiment """

        reward = pid_to_reward(x, self.target_context)
        # Build Data object
        return Data(
            df=pd.DataFrame(
                [{
                    "trial_index": trial_index,
                    "arm_name": arm_name,  # This is the default name used by Ax
                    "metric_name": "reward",
                    "mean": reward,
                    "sem": 0.0,  # Standard error of mean (can be improved later)
                }]
            )
        )
    def get_values(self, save_data):

        self.df = self.experiment.fetch_data().df

        self.df_optimization = self.df[self.df["trial_index"] >= self.limit]

        best_trial = self.df_optimization.sort_values("mean", ascending=False).iloc[0]
        #print(" --------- BEST TRIAL (after random):", best_trial['trial_index'], "with reward:", best_trial['mean'])

        best_trial_index = best_trial["trial_index"]
        self.best_arm = self.experiment.trials[best_trial_index].arm.parameters
        #print("Best PID parameters (after random):", self.best_arm)

        self

        self.trial_indices = self.df["trial_index"].tolist()
        self.rewards = self.df["mean"].tolist()
        self.best_trial_index = best_trial_index
        self.best_reward = best_trial['mean']

        if save_data:
            save_folder = "results"
            os.makedirs(save_folder, exist_ok=True)
            filling_str = "filling" if self.FILLING else f"train{self.train_context}" 
            filename = f"acrocbo_target{self.target_context}_{filling_str}_s{self.n_seed}it{self.n_iter}.csv"
            self.df.to_csv(os.path.join(save_folder, filename), index=False)

    def calculate_stats(self, th_optimum, optimum=None, vanilla_lowest = None):
        if optimum is None:
            optimum = min(self.rewards)
        
        regret = 0
        flag = 0
        count_bad = 0
        for i, mean in enumerate(self.rewards[self.limit:]):
            regret = regret + np.linalg.norm(mean-optimum)
            if mean < (optimum+th_optimum) and flag ==0:
                self.it_to_optimum = i+1
                flag = 1

            if vanilla_lowest is not None:
                if mean < vanilla_lowest:
                    count_bad =+ 1
        
        self.it_bad = count_bad if vanilla_lowest is not None else -1

        self.av_regret  = regret/(i+1)
        
        self.av_reward = stat.mean(self.rewards[self.limit:])
        self.dev_reward = np.sqrt(stat.variance(self.rewards[self.limit:], self.av_reward))
    
    def plot_data(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(self.trial_indices[self.limit:], self.rewards[self.limit:], marker='o', label = 'target opt')
        ax.plot(self.trial_indices[:self.limit], self.rewards[:self.limit], marker='o', label = 'filling space')
        
        ax.axvline(self.trial_indices[self.limit], color='black', linestyle='dotted')
        plt.xlabel("Trial Index")
        plt.ylabel("Mean Reward")
        plt.title(f"CBO for {self.target_context}\nMean: {self.av_reward:.2f}, Dev: {self.dev_reward:.2f}, Best: {self.best_reward:.2f}")

        save_folder = "results"
        os.makedirs(save_folder, exist_ok=True)
        filling_str = "filling" if self.FILLING else f"train{self.train_context}" 
        fig_filename = f"acrocbo_target{self.target_context}_{filling_str}_s{self.n_seed}it{self.n_iterations}_PLOT" + ".png"
        fig_save_path = os.path.join(save_folder, fig_filename)
        plt.savefig(fig_save_path)
        print(f"Figure saved as {fig_filename} in {save_folder}")

        plt.show()

        mean_values = np.array(self.rewards[self.limit:])
        plt.figure()
        plt.hist(mean_values, bins=np.arange(mean_values.min(), mean_values.max() + 5, 5), edgecolor='black')
        plt.xlabel('Mean Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Mean Values (intervals of 5)')

        fig_filename = f"acrocbo_target{self.target_context}_{filling_str}_s{self.n_seed}it{self.n_iterations}_HIST" + ".png"
        fig_save_path = os.path.join(save_folder, fig_filename)
        plt.savefig(fig_save_path)
        print(f"Figure saved as {fig_filename} in {save_folder}")

        plt.show()

class DummyRunner(Runner):
    def run(self, trial):
        return {} 
    