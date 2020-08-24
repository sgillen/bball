import gym
import numpy as np
from numpy import pi
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from multiprocessing import Process
from stable_baselines.common import make_vec_env
import os
import time
from bball3_env import BBall3Env
import pickle
import torch
import signal

num_steps = int(5e5)
base_dir = os.path.dirname(__file__) + "/data_ppo_wc/"
trial_name = input("Trial name: ")


def reward_fn(state, action):
    xpen = np.clip(-(state[3] - .15)**2, -1, 0)
    #xpen = 0.0

    ypen = np.clip(-(state[4] - 1)**2, -4, 0)
    #ypen = 0.0

    alive = 4.0
    return xpen + ypen + alive
  #  return -(state[4] - 1)**2 + alive

env_config = {
    'init_state': (0, 0, -pi / 2, .15, 1.2, 0, 0, 0, 0, 0),
    'reward_fn': reward_fn,
    'max_torque':  2.0,
    'max_steps' : 500
}

trial_dir = base_dir + trial_name + "/"
base_ok = input("run will be saved in " + trial_dir + " ok? y/n")

if base_ok == "n":
    exit()


def run_stable(num_steps, save_dir):

    env = make_vec_env(BBall3Env, n_envs=1, monitor_dir=save_dir, env_kwargs=env_config)

    model = PPO2(MlpPolicy,
                 env,
                 verbose=2,
                 seed=int(seed),
                 # normalize = True,
                 # policy = 'MlpPolicy',
                 n_steps=1024,
                 nminibatches=64,
                 lam=0.95,
                 gamma=0.99,
                 noptepochs=10,
                 ent_coef=0.0,
                 learning_rate=2.5e-4,
                 cliprange=0.1,
                 cliprange_vf=-1,
                 )

    num_epochs = 100

    for epoch in range(num_epochs):

        model.learn(total_timesteps=int(num_steps/num_epochs))
        model.save(save_dir + "/model.zip")


if __name__ == "__main__":

    start = time.time()
    proc_list = []

    os.makedirs(trial_dir, exist_ok=False)
    with open(trial_dir + "config.pkl", "wb") as config_file:
        pickle.dump(env_config, config_file)

    for seed in np.random.randint(0, 2 ** 32, 4):

        save_dir = trial_dir + "/" + str(seed)
        os.makedirs(save_dir, exist_ok=False)

        #run_stable(num_steps, save_dir)
        p = Process(
            target=run_stable,
            args=(num_steps, save_dir)
        )
        p.start()
        proc_list.append(p)

    for p in proc_list:
        print("joining")
        p.join()

    print(f"experiment complete, total time: {time.time() - start}, saved in {save_dir}")

