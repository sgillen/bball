import gym
import numpy as np
from numpy import pi
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from multiprocessing import Process
from stable_baselines.common import make_vec_env
import os
import time
from stable_baselines.common.vec_env.vec_normalize import VecNormalize


from bball3_env import BBall3Env
from bball3_mj_env import BBall3MJEnv
from bball3_pb_env import BBall3PBEnv

import pickle
import torch
import signal

num_steps = int(2e6)
base_dir = os.path.dirname(os.path.abspath(__file__)) + "/data_ppo/"
print(base_dir)
trial_name = input("Trial name: ")


slope_set = [-.01, 0.0, .01]
env_config = {"slope_set":slope_set, "random":True}

trial_dir = base_dir + trial_name + "/"
base_ok = input("run will be saved in " + trial_dir + " ok? y/n")

if base_ok == "n":
    exit()


def run_stable(num_steps, save_dir):
    env = make_vec_env(BBall3Env, n_envs=8, monitor_dir=save_dir, env_kwargs=env_config)
#    env = VecNormalize(env)

    model = PPO2(MlpPolicy,
                 env,
                 verbose=1,
                 seed=int(seed),
                 # normalize = True
                 # policy = 'MlpPolicy',
                 n_steps=2048,
                 nminibatches=32,
                 lam=0.95,
                 gamma=0.99,
                 noptepochs=10,
                 ent_coef=0.0,
                 learning_rate=2.5e-4,
                 cliprange=0.2,
                 cliprange_vf=-1,
                 )

    num_epochs = 5

    for epoch in range(num_epochs):

        model.learn(total_timesteps=int(num_steps/num_epochs))
        model.save(save_dir + "/model.zip")
#        env.save(save_dir + "/vec_env.zip")

 
if __name__ == "__main__":

    start = time.time()
    proc_list = []

    os.makedirs(trial_dir, exist_ok=False)
    with open(trial_dir + "config.pkl", "wb") as config_file:
        pickle.dump(env_config, config_file)

    for seed in np.random.randint(0, 2 ** 32, 1):

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

