
import numpy as np, pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt
import torch

from hebo.acquisitions.acq import EI_mine
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from hebo.optimizers.hebo_contextual import HEBO_VectorContextual
from gpytorch.kernels import RBFKernel, MaternKernel, ProductKernel, ScaleKernel
from gymnasium.envs.classic_control import AcrobotEnv
import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym")

# reproducibility

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

def control_acrobot(kp, ki, kd, epoch_num=5, target_height=1.0):
    env = MyAcrobotEnv(target_height, render_mode=None) 
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

def init_hebo_bigGP(n_seed = 10, n_iter = 20,  context_train =  'ctx_1.0'):
    space = DesignSpace().parse([
        {'name':'kp', 'type':'num', 'lb':-5, 'ub':5},
        {'name':'ki', 'type':'num', 'lb':-5, 'ub':5},
        {'name':'kd', 'type':'num', 'lb':-5, 'ub':5},
        {'name':'height', 'type':'num', 'lb':0.5, 'ub':1.5}
    ])

    opt = HEBO_VectorContextual(space, context_dict={
        'ctx_1.0': {'height': 1.0},
        'ctx_1.2': {'height': 1.2}
    },   rand_sample=n_seed*2)

    opt.context =  context_train

    x_init = []
    y_init = []

    # Sample only on context 1.0
    for i in range(n_seed):
        rec = opt.suggest(n=1)
        vals = rec.iloc[0]
        res = control_acrobot(vals['kp'], vals['ki'], vals['kd'], target_height=1.0)
        print(f"Sampling at ctx 1.0: Iter {i+1}, reward {res:.2f}")
        input_dict = vals.to_dict()
        input_dict['height'] = 1.0
        x_init.append(input_dict)
        y_init.append(-res)
        
    x_init = pd.DataFrame(x_init)
    y_init = np.array(y_init).reshape(-1,1)

    opt.observe_new_data(x_init, y_init)

    # add one more context for stability
    opt.context =  'ctx_1.2'
    x_init = []
    y_init = []

    for i in range(n_seed):
        rec = opt.suggest(n=1)
        vals = rec.iloc[0]
        res = control_acrobot(vals['kp'], vals['ki'], vals['kd'], target_height=1.2)
        print(f"Sampling at ctx 1.0: Iter {i+1}, reward {res:.2f}")
        input_dict = vals.to_dict()
        input_dict['height'] = 1.2
        x_init.append(input_dict)
        y_init.append(-res)
        
    x_init = pd.DataFrame(x_init)
    y_init = np.array(y_init).reshape(-1,1)

    opt.observe_new_data(x_init, y_init)

    return opt


def init_hebo_smallGP(n_seed = 5, n_iter = 5,  context_train =  'ctx_1.0'):
    # create big GP
    space = DesignSpace().parse([
        {'name':'kp', 'type':'num', 'lb':-5, 'ub':5},
        {'name':'ki', 'type':'num', 'lb':-5, 'ub':5},
        {'name':'kd', 'type':'num', 'lb':-5, 'ub':5},
        {'name':'height', 'type':'num', 'lb':0.5, 'ub':1.5}
    ])

    optBig = HEBO_VectorContextual(space, context_dict={
        'ctx_1.0': {'height': 1.0},
        'ctx_1.2': {'height': 1.2},
        'ctx_0.8': {'height': 0.8},
        'ctx_0.6': {'height': 0.6},
        'ctx_1.4': {'height': 1.4},
    },   rand_sample=0)

    optBig.hebo.acq_cls = EI_mine

    # create small GP
    space = DesignSpace().parse([
        {'name':'kp', 'type':'num', 'lb':-5, 'ub':5},
        {'name':'ki', 'type':'num', 'lb':-5, 'ub':5},
        {'name':'kd', 'type':'num', 'lb':-5, 'ub':5}
    ])

    optSmall = HEBO(space, rand_sample=n_seed, acq_cls = EI_mine)

    # run on small GP, for context = 0.8
    x_init = []
    y_init = []

    print('Populate with optimal for 1 case')
    for i in range(n_seed + n_iter):
        rec = optSmall.suggest(n_suggestions=1) 
        vals = rec.iloc[0]
        res = control_acrobot(vals['kp'], vals['ki'], vals['kd'], target_height=0.8)
        optSmall.observe(rec, np.array([[-res]]))
        print(f"Iter {i+1}: pid={vals.values.tolist()}, reward={res:.2f}")
        input_dict = vals.to_dict()
        input_dict['height'] = 0.8
        x_init.append(input_dict)
        y_init.append(-res)

    x_init = pd.DataFrame(x_init)
    y_init = np.array(y_init).reshape(-1,1)
    optBig.observe_new_data(x_init, y_init)

    # add one more context for stability
    # create small GP
    space = DesignSpace().parse([
        {'name':'kp', 'type':'num', 'lb':-5, 'ub':5},
        {'name':'ki', 'type':'num', 'lb':-5, 'ub':5},
        {'name':'kd', 'type':'num', 'lb':-5, 'ub':5}
    ])

    optSmall = HEBO(space, rand_sample=n_seed, acq_cls = EI_mine)

    # run on small GP, for context = 1.2
    x_init = []
    y_init = []

    print('Populate with optimal for 1.2 case')
    for i in range(n_seed+ n_iter):
        rec = optSmall.suggest(n_suggestions=1)
        vals = rec.iloc[0]
        res = control_acrobot(vals['kp'], vals['ki'], vals['kd'], target_height=1.2)
        print(f"Iter {i+1}: pid={vals.values.tolist()}, reward={res:.2f}")
        optSmall.observe(rec, np.array([[-res]]))
        input_dict = vals.to_dict()
        input_dict['height'] = 1.2
        x_init.append(input_dict)
        y_init.append(-res)
        
    x_init = pd.DataFrame(x_init)
    y_init = np.array(y_init).reshape(-1,1)

    optBig.observe_new_data(x_init, y_init)

    return optBig


def pid_optimize_hebo(opt, n_iter = 20, context = 'ctx_1.0'):

    opt.context = context

    print('------Start Optimization-----')
    for i in range(n_iter):
        rec = opt.suggest(n=1)
        vals = rec.iloc[0]
        res = control_acrobot(vals['kp'], vals['ki'], vals['kd'], target_height=opt.context_dict[opt.context]['height'])
        print(f"BO iter {i+1}: reward {res:.2f}")
        rec['height'] = opt.context_dict[opt.context]['height']
        opt.observe(rec, np.array([[-res]]))
    
    conv_bo_seq =  np.maximum.accumulate(-opt.hebo.y)
    
    best_idx = opt.hebo.y.argmin()
    best = opt.hebo.X.iloc[best_idx]
    print("Best PID:", best.to_dict(), "with reward", -opt.hebo.y[best_idx])

    return opt, best,  -opt.hebo.y


if __name__=='__main__':
    opt = init_hebo_smallGP()

    results = {}
    for ctx in ['ctx_1.0', 'ctx_1.4', 'ctx_0.6']:
        opt, best,y_evolution = pid_optimize_hebo(opt, context=ctx)
        results[ctx] =  y_evolution
        # Plot at the end
        fig = plt.figure()
        ax = fig.add_subplot()
        for i, (ctx, y_evolution) in enumerate(results.items()):
            ax.plot(y_evolution, marker='x', label=f'evolution {ctx}')
            ax.axvline((i+1) * 20, color='red', linestyle='dotted')
        ax.axvline(10, color='red', linestyle='dotted')
        ax.axvline(20, color='red', linestyle='dotted')
        plt.legend()
        plt.show()
