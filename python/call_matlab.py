# %%
from math import pi
from bball3_env import BBall3Env


def reward_fn(state, action):
    return state[3]


def done_criteria(state):
    return state[4] < 0


env_config = {
    'init_state': (0, 0, -pi / 2, .15, .75, 0, 0, 0, 0, 0),
    'reward_fn' : reward_fn,
}


env = BBall3Env(**env_config)
env.reset()

for _ in range(50):
    obs, reward, done, obs_dict = env.step([-1,3,3])
    env.animate(obs_dict["tout"], obs_dict["xout"])
    if done:
        break

env.reset()
