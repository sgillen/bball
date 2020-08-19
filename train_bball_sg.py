import gym
import seagul.envs

import numpy as np
from numpy import pi
from seagul.rl.ars.ars_pipe2 import ARSAgent
from seagul.nn import MLP
import torch
import pickle
import os

# init policy, valuefn
input_size = 10
output_size = 3
layer_size = 0
num_layers = 0
env_name = "bball3-v0"

num_steps = int(1e5)
base_dir = "data_ars/"
trial_name = input("Trial name: ")
torch.set_default_dtype(torch.float32)


def reward_fn(state, action):
    xpen = np.clip(-(state[4] - .45)**2, -1, 0)
    #ypen = np.clip(-(state[5] - 2)**2, -1, 0)
    ypen = 0.0
    alive = 2.0
    return xpen + ypen + alive


env_config = {
    'init_state': (0, 0, -pi / 2, .15, .75, 0, 0, 0, 0, 0),
    'reward_fn': reward_fn,
    'max_torque':  1.0
}

trial_dir = base_dir + trial_name + "/"
base_ok = input("run will be saved in " + trial_dir + " ok? y/n")

if base_ok == "n":
    exit()

os.makedirs(trial_dir, exist_ok=False)
with open(trial_dir + "config.pkl", "wb") as config_file:
    pickle.dump(env_config, config_file)


for seed in np.random.randint(0, 2**32, 4):
    policy = MLP(10,3,0,0)

    alg_config = {
        "env_name": env_name,
        "seed": int(seed),  # int((time.time() % 1)*1e8),
        "env_config" : env_config,
        "policy" : policy
    }

    agent = ARSAgent(**alg_config)
    agent.learn(int(1e2))

    torch.save(agent, f"{trial_dir}{seed}.agent")

