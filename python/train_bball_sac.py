import gym
import numpy as np
from numpy import pi
from seagul.rl.sac import sac
from seagul.nn import MLP
import torch
import pickle
import os
from bball3_env import BBall3Env

from torch.multiprocessing import Pool

from multiprocessing import Process
import seagul.envs

from seagul.rl.run_utils import run_sg
from seagul.rl.sac import SACAgent, SACModel
from seagul.nn import MLP

import torch
import torch.nn as nn
import numpy as np
from numpy import pi
import inspect

base_dir = os.path.dirname(__file__) + "/data_sac/"


def reward_fn(state, action):
    xpen = np.clip(-(state[3] - .15)**2, -1, 0)
    #xpen = 0.0

    ypen = np.clip(-(state[4] - 1.2)**2, -4, 0)
    #ypen = 0.0

    alive = 5.0
    return xpen + ypen + alive

def run_and_save(arg):
    seed, save_dir = arg
    trial_dir = save_dir + "/" + "seed" + str(seed) + "/"

    # init policy, value fn
    input_size = 10
    output_size = 3
    layer_size = 32
    num_layers = 2
    activation = nn.ReLU
    env_name = "bball3-v1"
    num_steps = int(1e5)

    policy = MLP(input_size, output_size*2, num_layers, layer_size, activation)
    value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
    q1_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation)
    q2_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation)

    env_config = {
        'init_state': (0, 0, -pi / 2, .15, .75, 0, 0, 0, 0, 0),
        'reward_fn': reward_fn,
        'max_torque': 5.0,
        'max_steps': 50
    }

    model = SACModel(
        policy=policy,
        value_fn = value_fn,
        q1_fn = q1_fn,
        q2_fn = q2_fn,
        act_limit = env_config['max_torque'],
    )


    alg_config = {
        "env_name": env_name,
        "model": model,
        "alpha": .05,
        "seed": seed,
        "exploration_steps": 1000,
        "min_steps_per_update": 500,
        "gamma": 1,
        "sgd_batch_size": 128,
        "replay_batch_size": 512,
        "iters_per_update": 16,
        # "iters_per_update": float('inf'),
        "env_config": env_config
    }

    agent = SACAgent(**alg_config)
    agent.learn(num_steps)

    os.makedirs(trial_dir, exist_ok=False)

    with open(trial_dir + "agent.ag", "wb") as agent_file:
        torch.save(agent, agent_file)

    with open(trial_dir + "config.pkl", "wb") as config_file:
        pickle.dump(env_config, config_file)

    with open(trial_dir + "reward_fn.py", 'w') as f:
        f.write(inspect.getsource(reward_fn))




if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    seeds = np.random.randint(0, 2**32, 8)
    pool = Pool(processes=8)

    trial_name = input("Trial name: ")
    trial_dir = base_dir + trial_name + "/"
    input("run will be saved in " + trial_dir + " ok? Ctrl-C to cancel")

    seeds_and_dirs = [(seed, trial_dir) for seed in seeds]
    # results = run_and_test(run_and_test(seeds[0]))
    pool.map(run_and_save, seeds_and_dirs)
