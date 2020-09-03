import gym
from gym import register
import numpy as np
import pybullet as p
import pybullet_data
from pybullet_envs.env_bases import MJCFBaseBulletEnv
import os


class BBall3PBEnv(gym.Env):
    motor_joints = [0, 2, 4]
    num_joints = 3

    def __init__(self,
                 render=False,
                 init_noise=.005,
                 num_steps=500,
                 torque_limits=[50] * 3,
                 physics_params=None,
                 dynamics_params=None,
                 ):

        self.args = locals()

        self.cur_step = 0
        self.num_steps = num_steps

        self.init_noise = init_noise
        self.num_states = 10

        self.cur_step = 0
        self.torque_limits = np.array(torque_limits)

        low = -np.ones(6)
        self.action_space = gym.spaces.Box(low=low, high=-low, dtype=np.float32)

        low = -np.ones(17) * np.inf
        self.observation_space = gym.spaces.Box(low=low, high=-low, dtype=np.float32)

        self.init_noise = init_noise

        self.cur_step = 0

        low = -np.ones(3)
        self.action_space = gym.spaces.Box(low=low, high=-low, dtype=np.float32)

        low = -np.ones(10) * np.inf
        self.observation_space = gym.spaces.Box(low=low, high=-low, dtype=np.float32)

        if render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

            # self.plane_id = p.loadSDF(pybullet_data.getDataPath() + "/plane_stadium.sdf")[0]
        self.world_id, self.arm_id, self.ball_id = p.loadMJCF('/home/sgillen/work/bball/python/assets/bball3.xml', flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)

        self.reset()

    def reset(self):

        p.removeBody(self.world_id)
        p.removeBody(self.arm_id)
        p.removeBody(self.ball_id)

        # self.plane_id = p.loadSDF(pybullet_data.getDataPath() + "/plane_stadium.sdf")[0]
        self.world_id, self.arm_id, self.ball_id = p.loadMJCF('/home/sgillen/work/bball/python/assets/bball3.xml', flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)

        def set_motors_zero(body_id):
            p.setJointMotorControlArray(body_id,
                                        [i for i in range(p.getNumJoints(body_id))],
                                        p.POSITION_CONTROL,
                                        # positionGain=0.1,
                                        # velocityGain=0.1,
                                        forces=[0 for _ in range(p.getNumJoints(body_id))]
                                        )

        # world_id, arm_id, ball_id = p.loadMJCF('/home/sgillen/work/bball/python/assets/bball3.xml')
        p.setGravity(0, -9.8, 0.0)
        #p.resetBasePositionAndOrientation(self.ball_id, [0, 1.2, 0], [0, 0, 0, 1.0])
        p.resetJointState(self.arm_id, 4, -np.pi / 4)
        set_motors_zero(self.arm_id)
        set_motors_zero(self.ball_id)
        self.cur_step = 0

        return self._get_obs()

    def step(self, a):
        a = np.clip(a, -1, 1)

        # forces = (a*np.array([40, 40, 12, 40, 40, 12])).tolist()
        # forces = a.tolist()
        # forces = (a*np.array([100, 100, 100, 100, 100, 100])).tolist()
        forces = (a * self.torque_limits).tolist()

        p.setJointMotorControlArray(self.arm_id, self.motor_joints, p.TORQUE_CONTROL, forces=forces)
        p.stepSimulation()

        states = self._get_obs()

        xpen = np.clip(-(states[3] - .3) ** 2, -1, 0)
        ypen = np.clip(-(states[4] - 1.5) ** 2, -4, 0)
        # import ipdb; ipdb.set_trace()
        apen = .1 * (-np.sum(np.abs(a)) ** 2)
        alive = 5.0
        reward = xpen + ypen + apen + alive

        h0 = .3 * np.cos(states[0])
        h1 = h0 + .3 * np.cos(states[1] + states[0])
        h2 = h1 + .3 * np.cos(states[2] + states[1] + states[0])
        self.cur_step += 1

        done = (not np.isfinite(states).all()) or states[4] < 0 or self.cur_step >= self.num_steps or h0 < 0 or h2 < 0 or h2 < 0

        if not (self.cur_step < self.num_steps):
            done = True

        return self._get_obs(), reward, done, {}


    def _get_obs(self):
        motor_joints = [0, 2, 4]
        arm_states = p.getJointStates(self.arm_id, motor_joints)
        ball_states = p.getJointStates(self.ball_id, [0, 1])

        pos = np.zeros(5)
        vel = np.zeros(5)

        for i in [0, 1, 2]:
            pos[i] = arm_states[i][0]
            vel[i] = arm_states[i][1]

        for i in [0, 1]:
            pos[i + 3] = ball_states[i][0]
            vel[i + 3] = ball_states[i][1]

        pos[4] += 1.2 # hack for ball height

        return np.concatenate((pos, vel))

    def render(self, *args):
        pass

    def close(self):
        p.removeBody(self.world_id)
        p.removeBody(self.arm_id)
        p.removeBody(self.ball_id)


register("bball3_pb-v0", entry_point=BBall3PBEnv)

    #self.walker_id = p.loadMJCF(pybullet_data.getDataPath() + "/mjcf/walker2d.xml")[0]
    #flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)[0] # TODO not sure the self collision needs to be here..


#
# class BBall3
#
#
# class BBall3BPEnv(MJCFBaseBulletEnv):
#     def __init__(self):
#         self.robot =
#         super().__init__(os.path.abspath(os.path.dirname(__file__)) + "/assets/bball3.xml",
#
#
#         def reset
