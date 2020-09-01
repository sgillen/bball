from stable_baselines import TD3
import gym
import numpy as np
from numpy import pi
#from stable_baselines.common.policies import MlpPolicy
from stable_baselines.td3.policies import MlpPolicy
from multiprocessing import Process
from stable_baselines.common import make_vec_env
import os
import time
import pickle
import torch
import signal
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from bball3_env import BBall3Env


num_steps = int(5e5)
base_dir = "data_ddpg_wc/"
trial_name = input("Trial name: ")


def reward_fn(state, action):
    xpen = np.clip(-(state[3] - .45)**2, -1, 0)
    #xpen = 0.0

    ypen = np.clip(-(state[4] - 2)**2, -4, 0)
    #ypen = 0.0

    alive = 5.0
    #return xpen + ypen + alive
    return -(state[4] - 2)**2 + alive

env_config = {
    'init_state': (0, 0, -pi / 2, .15, .75, 0, 0, 0, 0, 0),
    'reward_fn': reward_fn,
    'max_torque':  5.0
}

trial_dir = base_dir + trial_name + "/"
base_ok = input("run will be saved in " + trial_dir + " ok? y/n")

if base_ok == "n":
    exit()


def run_stable(num_steps, save_dir):
    env = make_vec_env(BBall3Env, n_envs=1, monitor_dir=save_dir, env_kwargs=env_config)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))

    model = TD3(MlpPolicy,
                env,
                action_noise=action_noise,
                verbose=1,
                gamma=0.99,
                buffer_size=1000000,
                learning_starts=10000,
                batch_size=100,
                learning_rate=1e-3,
                train_freq=1000,
                gradient_steps=1000,
                policy_kwargs={"layers": [64, 64]},
                n_cpu_tf_sess=1,
                )

    num_epochs = 1
    total_steps = 5e5

    for epoch in range(num_epochs):
        model.learn(total_timesteps=int(total_steps/num_epochs))
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


