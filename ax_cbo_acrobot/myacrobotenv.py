import gymnasium as gym
import numpy as np

from gymnasium.envs.classic_control import AcrobotEnv

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
    control_signal = kp * error + ki * integral + kd * derivative
    return control_signal, error, integral

def control_acrobot(kp, ki, kd, target, epoch_num=5):
    env = MyAcrobotEnv(target, render_mode=None) 
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

# calculate the reward
def pid_to_reward(pid, target):
    kp = pid['kp']
    ki = pid['ki']
    kd = pid['kd']
    mean_total_reward = control_acrobot(kp, ki, kd, target)
    return mean_total_reward
