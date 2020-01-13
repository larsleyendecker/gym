import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_custom_env
import json
import mujoco_py

class Ur5HegEnv(mujoco_custom_env.MujocoCustomEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        with open("/home/marius/DRL_SetBot-RearVentilation/experiment_configs/goal.json", encoding='utf-8') as file:
            self.target = json.load(file)
        mujoco_custom_env.MujocoCustomEnv.__init__(
            self, '/home/marius/DRL_SetBot-RearVentilation/UR5+gripper/UR5_position_control.xml', 2)


    def step(self, a):
        vec_pos = self.sim.data.get_body_xpos("gripper_dummy_heg")-self.target["xpos"]
        vec_quat = self.sim.data.get_body_xquat("gripper_dummy_heg")-self.target["xquat"]

        reward_dist = - np.linalg.norm(vec_pos) - np.linalg.norm(vec_quat)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = np.array([1.5708, 0, -1.5708, 1.5708, 1.5708, 1.5708])
        self.sim.data.ctrl[:] = qpos
        self.set_state(qpos, np.array([0, 0, 0, 0, 0, 0]))
        return self._get_obs()

    def _get_obs(self):
        force = self.sim.data.sensordata
        return np.concatenate([
            force,

            self.data.get_body_xpos("gripper_dummy_heg")-self.target["xpos"],
            self.data.get_body_xquat("gripper_dummy_heg")-self.target["xquat"]
        ])

