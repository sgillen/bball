from gym import register
import numpy as np
import os
from gym.envs.mujoco.mujoco_env import MujocoEnv


class BBall3MJEnv(MujocoEnv):
    def __init__(self, num_steps=500):
        self.cur_step = 0
        self.num_steps = num_steps

        super().__init__(os.path.abspath(os.path.dirname(__file__)) + "/assets/bball3.xml", 1)

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qpos[2] -= np.pi/4
        self.set_state(qpos, qvel)
        self.cur_step = 0
        return self.state_vector()

    def step(self, acts):
        self.do_simulation(acts, self.frame_skip)

        states = self.state_vector()

        xpen = np.clip(-(states[3] - .3)**2,     -1, 0)
        ypen = np.clip(-(states[4] - 1.5)**2,    -4, 0)
        #import ipdb; ipdb.set_trace()
        apen = .1*(-np.sum(np.abs(acts))**2)
        alive = 5.0
        reward = xpen + ypen + apen + alive

        h0 = .3 * np.cos(states[0])
        h1 = h0 + .3 * np.cos(states[1] + states[0])
        h2 = h1 + .3 * np.cos(states[2] + states[1] + states[0])
        self.cur_step += 1

        done = (not np.isfinite(states).all()) or states[4] < 0 or self.cur_step >= self.num_steps or h0 < 0 or h2 < 0 or h2 < 0

        return states, reward, done, {}

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        # self.viewer.cam.distance = 3.0
        # self.viewer.cam.lookat[0] = 0
        # self.viewer.cam.lookat[1] = 0.0
        # self.viewer.cam.lookat[2] = 5.0
        # self.vi


register("bball3_mj-v0", entry_point=BBall3MJEnv)