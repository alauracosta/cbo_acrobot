
import numpy as np, pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt

from hebo.acquisitions.acq import EI_mine
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from gymnasium.envs.classic_control import AcrobotEnv
import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym")


# reproducibility
SEED = 4
random.seed(SEED)
np.random.seed(SEED)

from hebo.optimizers.hebo import np as hebo_np
hebo_np.random.seed(SEED)



class MyAcrobotEnv(AcrobotEnv):
    def __init__(self, target_height=1.0, **kw):
        super().__init__(**kw)
        self.target_height = target_height
    def _terminal(self):
        s = self.state
        return bool(-np.cos(s[0]) - np.cos(s[1] + s[0]) > self.target_height)

def compute_pid_control(kp, ki, kd, target, current, prev_error, integral):
    error = target - current
    integral += error
    derivative = error - prev_error
    return kp*error + ki*integral + kd*derivative, error, integral

def control_acrobot(kp, ki, kd, target_height = 1.0, epoch_num=5):
    env = MyAcrobotEnv(target_height=target_height, render_mode=None) 
    rewards = []

    for seed in (20 + i for i in range(epoch_num)):
        obs, _ = env.reset(seed=seed)
        env.action_space.seed(seed)
        prev_error = integral = 0
        total = 0
        for _ in range(500):
            state = env.state
            cs, prev_error, integral = compute_pid_control(kp, ki, kd, 0, state[0], prev_error, integral)
            obs, reward, terminated, truncated, _ = env.step(max(min(int(cs), 2), 0))
            done = terminated or truncated
            total += reward
            if done:
                break
        rewards.append(total)
    env.close()
    return np.mean(rewards)

def pid_optimize_hebo(n_seed=10, n_iter=20, target_height = 1.0):
    # define HEBO search space
    space = DesignSpace().parse([
        {'name':'kp','type':'num','lb':-5,'ub':5},
        {'name':'ki','type':'num','lb':-5,'ub':5},
        {'name':'kd','type':'num','lb':-5,'ub':5},
    ])
    opt = HEBO(space, rand_sample=n_seed, acq_cls = EI_mine)

    # initial random + bayesopt iterations
    for i in range(n_seed + n_iter):
        rec = opt.suggest(n_suggestions=1)  # DataFrame with kp,ki,kd
        vals = rec.iloc[0]
        res = control_acrobot(vals['kp'], vals['ki'], vals['kd'], target_height)
        opt.observe(rec, np.array([[-res]]))
        print(f"Iter {i+1}: pid={vals.values.tolist()}, reward={res:.2f}")

    conv_bo_seq =  np.maximum.accumulate(-opt.y)
    # extract best
    best_idx = opt.y.argmin()
    best = opt.X.iloc[best_idx]
    print("Best PID:", best.to_dict(), "with reward", -opt.y[best_idx])

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(conv_bo_seq, marker='o', label='convergence')
    ax.plot(-opt.y, marker='x', label='evolution')
    ax.axvline(n_seed, color='red', linestyle='dotted')
    plt.legend()
    plt.show()

    return opt, best


if __name__=='__main__':
    opt, best = pid_optimize_hebo(n_seed=10, target_height=1.4)