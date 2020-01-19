import os
from gym import utils
from gym.envs.robotics import ur5_env
import numpy as np
import os


# Ensure we get the path separator correct on windows
HOME_PATH = os.getenv("HOME")
MODEL_XML_PATH = os.path.join(*[HOME_PATH, "DRL_SetBot-RearVentilation", "UR5+gripper", "UR5_position_control.xml"])
initial_qpos = np.array([1.5708, 0, -1.5708, 1.5708, 1.5708, 1.5708])

class Ur5HegEnv(ur5_env.Ur5Env, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        ur5_env.Ur5Env.__init__(
            self, MODEL_XML_PATH, n_substeps=1, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
