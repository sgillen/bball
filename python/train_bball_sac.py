import gym
import numpy as np
from numpy import pi
from seagul.rl.sac import sac
from seagul.nn import MLP
import torch
import pickle
import os
from bball3_env import BBall3Env

from multiprocessing import Process
import seagul.envs

from seagul.rl.run_utils import run_sg
from seagul.rl.sac import sac, SACModel
from seagul.nn import MLP
from seagul.integration import euler

import torch
import torch.nn as nn
import numpy as np
from numpy import pi



def reward_fn(state, action):
    xpen = np.clip(-(state[3] - .15)**2, -1, 0)
    #xpen = 0.0

    ypen = np.clip(-(state[4] - 1.2)**2, -4, 0)
    #ypen = 0.0

    alive = 5.0
    return xpen + ypen + alive

env_config = {
    'init_state': (0, 0, -pi / 2, .15, 1.2, 0, 0, 0, 0, 0),
    'reward_fn': reward_fn,
    'max_torque': 5.0,
    'max_steps': 500
}

# init policy, value fn
input_size = 10
output_size = 3
layer_size = 32
num_layers = 2
activation = nn.ReLU
env_name = "bball3-v1"

if __name__ == "__main__":

    num_steps = int(1e5)
    base_dir = os.path.dirname(__file__) + "/data_sac/"
    trial_name = input("Trial name: ")
    torch.set_default_dtype(torch.float32)

    trial_dir = base_dir + trial_name + "/"
    base_ok = input("run will be saved in " + trial_dir + " ok? y/n")

    if base_ok == "n":
        exit()

    os.makedirs(trial_dir, exist_ok=False)
    with open(trial_dir + "config.pkl", "wb") as config_file:
        pickle.dump(env_config, config_file)

    proc_list = []

    for seed in np.random.randint(0, 2 ** 32, 4):

        policy = MLP(input_size, output_size*2, num_layers, layer_size, activation)
        value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
        q1_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation)
        q2_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation)


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
            "seed": seed,  # int((time.time() % 1)*1e8),
            "train_steps" : 5e4,
            "alpha" : .05,
            "exploration_steps" : 1000,
            "min_steps_per_update" : 500,
            "gamma": 1,
            "sgd_batch_size": 128,
            "replay_batch_size" : 4096,
            "iters_per_update": 16,
            #"iters_per_update": float('inf'),
            "env_config": env_config
        }

        p = Process(
            target=run_sg,
            args=(alg_config, sac, "sac-test", "", base_dir + trial_name + "/" + "seed" + str(seed)),
        )
        p.start()
        proc_list.append(p)

    for p in proc_list:
        print("joining")
        p.join()
