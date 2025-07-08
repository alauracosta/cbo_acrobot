import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from ax import Metric
from ax import (
    RangeParameter, ParameterType, SearchSpace, OptimizationConfig,
    Objective, Experiment, Data, Models
)
from ax.core.arm import Arm
from ax.core.runner import Runner
from ax.core.metric import Metric
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import rbf_kernel
from scipy.optimize import minimize

# Environment setup
env_name = "Acrobot-v1"
tkwargs = {"dtype": torch.double, "device": "cpu"}
device = tkwargs["device"]


# GP model for 
class GPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())
    def forward(self, x):
        return self.mean_module(x), self.covar_module(x)

class DummyRunner(Runner):
    def run(self, trial):
        return {}

class AcrobotSimulator:
    def __init__(self):
        self.env = gym.make(env_name)
        self.max_steps = 500
    def reset(self, seed=0):
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        return self.env.state.copy()
    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        return self.env.state.copy(), reward, done
    def close(self):
        self.env.close()

def compute_pid_control(kp, ki, kd, target, current, prev_error, integral):
    error = target - current
    integral += error
    derivative = error - prev_error
    control_signal = kp * error + ki * integral + kd * derivative
    return control_signal, error, integral

def collect_real_data(pid, seed=0, steps=500):
    kp = pid['kp']
    ki = pid['ki']
    kd = pid['kd']
    sim = AcrobotSimulator()
    state = sim.reset(seed)
    traj = []
    total_reward = 0
    prev_error, integral = 0, 0
    for _ in range(steps):
        control_signal, prev_error, integral = compute_pid_control(kp, ki, kd, 0, state[0], prev_error, integral)
        action = np.clip(int(control_signal), 0, 2)
        next_state, reward, done = sim.step(action)
        traj.append((state.copy(), action, reward, next_state.copy()))
        total_reward += reward
        state = next_state
        if done:
            break
    sim.close()
    return traj, total_reward


def simulate_with_model(pid, models, steps=500):
    kp, ki, kd = pid['kp'], pid['ki'], pid['kd']
    prev_error, integral = 0, 0
    state = np.zeros(4)
    total_reward = 0
    for _ in range(steps):
        control_signal, prev_error, integral = compute_pid_control(kp, ki, kd, 0, state[0], prev_error, integral)
        action = np.clip(int(control_signal), 0, 2)
        input_vec = np.append(state, action).reshape(1, -1)
        reward_pred = models['reward'].predict(input_vec)[0]
        next_state_pred = [models['dynamics'][i].predict(input_vec)[0] for i in range(4)]
        state = next_state_pred
        total_reward += reward_pred
    return total_reward


def train_models(trajectories):
    X = []
    Y_next = [[] for _ in range(4)]
    Y_reward = []
    for traj in trajectories:
        for s, a, r, s_next in traj:
            feat = np.append(s, a)
            X.append(feat)
            for i in range(4):
                Y_next[i].append(s_next[i])
            Y_reward.append(r)
    X = np.array(X)
    models = {
        'dynamics': [LinearRegression().fit(X, np.array(Y_next[i])) for i in range(4)],
        'reward': LinearRegression().fit(X, Y_reward)
    }
    return models

# === Main MBOA Class ===
class MBOA:
    def __init__(self, n_seed=5, n_iter=15):
        self.n_seed = n_seed
        self.n_iter = n_iter
        self.trajectories = []
        self.experiment = None
        self.models = None
        self.data = None

    def setup_experiment(self):
        search_space = SearchSpace(parameters=[
            RangeParameter(name="kp", lower=-5.0, upper=5.0, parameter_type=ParameterType.FLOAT),
            RangeParameter(name="ki", lower=-5.0, upper=5.0, parameter_type=ParameterType.FLOAT),
            RangeParameter(name="kd", lower=-5.0, upper=5.0, parameter_type=ParameterType.FLOAT),
        ])
        self.experiment = Experiment(
            name="mboa_acrobot",
            search_space=search_space,
            optimization_config=OptimizationConfig(
                objective=Objective(metric=Metric('reward'), minimize=False),
            ),
            runner=DummyRunner(),
        )

    def init_data(self):
        sobol = Models.SOBOL(search_space=self.experiment.search_space)
        for _ in range(self.n_seed):
            gr = sobol.gen(1)
            trial = self.experiment.new_trial(generator_run=gr)
            trial.runner = self.experiment.runner
            trial.mark_running()
            arm = trial.arm
            traj, reward = collect_real_data(arm.parameters)
            self.trajectories.append(traj)
            data = Data(df=pd.DataFrame([{
                "trial_index": trial.index,
                "arm_name": arm.name,
                "metric_name": "reward",
                "mean": reward,
                "sem": 0.0,
            }]))
            self.experiment.attach_data(data)
            trial.mark_completed()
        self.data = self.experiment.fetch_data()

    def optimize(self):
        print('Building the surrogate models')
        self.models = train_models(self.trajectories)
        for i in range(self.n_iter):
            print(f"\n[{i+1}/{self.n_iter}]")

            # Data base of all the policies tested and respective returns
            self.policies = [list(trial.arm.parameters.values()) for trial in self.experiment.trials.values()]
            self.returns = np.array([row['mean'] for _, row in self.data.df.iterrows()])
            theta_array = np.array(self.policies)

            # m(theta, D) vector - model prediction of the returns for each policy
            model_preds = np.array([simulate_with_model({'kp': p[0], 'ki': p[1], 'kd': p[2]}, self.models) for p in self.policies])
            K = rbf_kernel(theta_array, theta_array, gamma=0.1)

            # eta(theta) - actual real returns for each policy 
            y = self.returns.reshape(-1, 1)
            m = model_preds.reshape(-1, 1)
            K_inv = np.linalg.inv(K + 1e-6 * np.eye(len(K)))
            
            # calculate beta based on the gradient of the log-likelihood
            beta = (y.T @ K_inv @ m) / (m.T @ K_inv @ m)
            beta = beta.item()

            print(f"  -> Optimized beta: {beta:.4f}")

            # GP with a prior mean of beta*m
            residuals = self.returns - beta * model_preds

            def acquisition(theta):
                theta = np.array(theta).reshape(1, -1)
                k_star = rbf_kernel(theta, theta_array, gamma=0.1)
                gp_corr = k_star @ K_inv @ (self.returns - beta * model_preds)
                # get the m(theta, D) for the new theta
                m_theta = simulate_with_model({'kp': theta[0, 0], 'ki': theta[0, 1], 'kd': theta[0, 2]}, self.models)
                return -(beta * m_theta + gp_corr[0])
            
            res = minimize(acquisition, x0=np.random.uniform(-5, 5, 3), bounds=[(-5, 5)] * 3, method='L-BFGS-B')

            # parameters that minimizes acq function
            best_params = {'kp': res.x[0], 'ki': res.x[1], 'kd': res.x[2]}
            print(f"  -> Selected next policy: {best_params}")

            # simulate the chosen parameters in the real environment
            traj, reward = collect_real_data(best_params)
            print(f"  -> Reward: {reward}")
            self.trajectories.append(traj)
            trial = self.experiment.new_trial()
            trial.runner = self.experiment.runner
            trial.add_arm(Arm(parameters=best_params))
            trial.mark_running()
            trial.mark_completed()
            new_data = Data(df=pd.DataFrame([{
                "trial_index": trial.index,
                "arm_name": trial.arm.name,
                "metric_name": "reward",
                "mean": reward,
                "sem": 0.0,
            }]))
            self.experiment.attach_data(new_data)
            self.data = self.experiment.fetch_data()

    def best_result(self):
        df = self.experiment.fetch_data().df
        best = df.sort_values("mean", ascending=False).iloc[0]
        print("Best Trial:", best['trial_index'], "Reward:", best['mean'])
        print("Best PID:", self.experiment.trials[best['trial_index']].arm.parameters)
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(self.data.df['trial_index'], self.data.df['mean'], marker='o')
        plt.show()

# === Run the optimizer ===
if __name__ == "__main__":
    mboa = MBOA()
    print('Setting up experiment')
    mboa.setup_experiment()
    print('Random Sampling')
    mboa.init_data()
    print('Start Optimization')
    mboa.optimize()
    mboa.best_result()












