import os
from gym import utils
from gym.envs.robotics import ur10_env
from gym.envs.robotics import ur10_rel_env

import numpy as np
import numpy
import os


# Ensure we get the path separator correct on windows
HOME_PATH = os.getenv("HOME")
MODEL_XML_PATH = os.path.join(*[HOME_PATH, "DRL_SetBot-RearVentilation", "UR10", "ur10_heg.xml"])
initial_qpos = numpy.array([1.5708, -1.3, 2.1, -0.80, 1.5708, 3.14159])

class Ur10HegEnv(ur10_env.Ur10Env, utils.EzPickle):
    def __init__(self, reward_type='dense'):
        ur10_env.Ur10Env.__init__(
            self, MODEL_XML_PATH, n_substeps=2, distance_threshold=0.035,
            initial_qpos=initial_qpos, reward_type=reward_type, ctrl_type='cartesian')
        utils.EzPickle.__init__(self)

class Ur10HegSparseEnv(ur10_env.Ur10Env, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        ur10_env.Ur10Env.__init__(
            self, MODEL_XML_PATH, n_substeps=2, distance_threshold=0.035,
            initial_qpos=initial_qpos, reward_type=reward_type, ctrl_type='cartesian')
        utils.EzPickle.__init__(self)

class Ur10HegRelEnv(ur10_rel_env.Ur10Env, utils.EzPickle):
    def __init__(self, reward_type='dense'):
        ur10_rel_env.Ur10Env.__init__(
            self, MODEL_XML_PATH, n_substeps=2, distance_threshold=0.01,
            initial_qpos=initial_qpos, reward_type=reward_type, ctrl_type='cartesian')
        utils.EzPickle.__init__(self)