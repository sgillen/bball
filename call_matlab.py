# %%
from seagul.envs.matlab import BBallEnv, BBall3Env

env = BBallEnv()
env.reset()


for _ in range(10):
    obs, reward, done, obs_dict = env.step([4,4])

env.reset()
env.animate(obs_dict["tout"], obs_dict["xout"])

env3 = BBall3Env()
obs, reward, done, obs_dict = env3.step([4, 4, 4])
env3.animate(obs_dict["tout"], obs_dict["xout"])
