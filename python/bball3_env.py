from gym.envs.registration import register
from gym import core, spaces
from gym.utils import seeding
import matlab.engine
import matlab
import numpy as np
from numpy import pi
import seagul
import seagul.envs.matlab.bball3_src
import os

class BBall3Env(core.Env):
    """ Connects to nihars bouncing ball matlab src

    states are
    [0] lower link angle (rad) (absolute?)
    [1] middle link angle (rad) (relative?)
    [2] middle link angle (rad) (relative?)
    [3] ball x position (m)
    [4] ball y position (m)
    [5] lower link velocity (rad/s)
    [6] middle link velocity (rad/s)
    [7] upper link velocity (rad/s)
    [8] ball x velocity (m/s)
    [9] ball y velocity (m/s)
    """

    def __init__(self,
                 max_torque=5.0,
                 dt=.02,
                 seed=None,
                 init_state=(-pi/4, 0.0, -3*pi/4, 0.025, .5, 0, 0, 0, 0, 0),
                 init_state_weights=(pi, pi, pi, .5, .5, 0, 0, 0, 0, 0),
                 reward_fn=lambda s, a: s[4],
                 max_steps = 100,
                 ):

        print("before engine")
        self.eng = matlab.engine.start_matlab()
        print("got here")
        self.eng.addpath(os.path.dirname(__file__) + "/../matlab/", nargout=0)
        self.max_torque = max_torque
        self.dt = dt
        self.init_state = matlab.single(init_state, size=(10, 1))
        self.init_state_weights = matlab.single(init_state_weights, size=(10, 1))
        np.random.seed(seed)

        self.reward_fn = reward_fn

        low = np.array([-pi, -pi, -pi, -5, -5, -10, -30, -30, -10, -10])
        self.observation_space = spaces.Box(low=low, high=-low, dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-max_torque, -max_torque, -max_torque]), high=np.array([max_torque, max_torque, max_torque]), dtype=np.float32)

        self.impact_miss = 0

        self.max_steps = max_steps
        self.cur_step = 0
        self.reset()

    def reset(self):
        self.t = 0
        init_state = self.init_state
        self.cur_step = 0
        #init_state += self.eng.rand(8)*(self.init_state_weights*matlab.single([2.0])) - self.init_state_weights

        self.state = init_state
        return np.array(init_state, dtype=np.float32).reshape((10,))

    def step(self, action):
        action = np.clip(action, -self.max_torque, self.max_torque)
        action = matlab.single(action.tolist())
        action.reshape((3, 1))
        tout, xout = self.eng.integrateODE(matlab.single([self.t, self.t + self.dt]), self.state, self.dt, action, nargout=2)

        impactState, impactTime = self.eng.detectImpact(tout, xout, nargout=2)

        if np.array(xout)[-1,-1] >= 0 and impactTime != -1:
            self.impact_miss += 1
            impactTime = -1
            print(f"Impact miss, total for this env: {self.impact_miss}")
            #input()

        if impactTime == -1:  # No contact
            self.state = matlab.single(xout[-1])
            self.state.reshape((10, 1))
            # self.state.reshape((8,1))
            self.t = np.array(tout[-1]).item()
        else:
            #print("contact!")
            #print(impactState)
            impactState.reshape((10, 1))
            self.state = self.eng.impact(matlab.single(impactState))
            # self.state.reshape((8,1))
            self.t = np.array(impactTime).item()

        reward = self.reward_fn(np.array(self.state), action)
        done = self.eng.constraint(self.state)

        # if done:
        #     print(done)


        self.cur_step += 1
        if self.cur_step > self.max_steps:
            done = True

        return np.array(self.state, dtype=np.float32).reshape((10,)), reward.item(), done, {"tout": tout, "xout": xout}

    def render(self):
        raise NotImplementedError('Frame by frame rendering not supported, call animate instead')

    def animate(self, t, x):
        self.eng.animate(t, x, nargout=0)
    def close(self):
        self.eng.quit()


register("bball3-v1", entry_point=BBall3Env)
