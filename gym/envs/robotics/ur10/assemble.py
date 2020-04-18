import os
from gym import utils
from gym.envs.robotics import ur10_env
from gym.envs.robotics import ur10_rel_env
from gym.envs.robotics import ur10_rel_simpheg_env
from gym.envs.robotics import ur10_corrective_env
from gym.envs.robotics import ur10_noisy_pd_env



import numpy as np
import numpy
import os


# Ensure we get the path separator correct on windows
HOME_PATH = os.getenv("HOME")
MODEL_XML_PATH = os.path.join(*[HOME_PATH, "DRL_SetBot-RearVentilation", "UR10", "ur10_heg.xml"])
MODEL_XML_PATH_SLOW = os.path.join(*[HOME_PATH, "DRL_SetBot-RearVentilation", "UR10", "ur10_heg_slow.xml"])
MODEL_XML_PATH_SLOW_SH = os.path.join(*[HOME_PATH, "DRL_SetBot-RearVentilation", "UR10", "ur10_heg_slow_simpheg.xml"])
MODEL_XML_PATH_SLOW_SH_CONF2 = os.path.join(*[HOME_PATH, "DRL_SetBot-RearVentilation", "UR10", "ur10_heg_slow_simpheg_conf2.xml"])
MODEL_XML_PATH_RAND = os.path.join(*[HOME_PATH, "DRL_SetBot-RearVentilation", "UR10", "ur10_assembly_setup_rand_temp_1.xml"])


initial_qpos = numpy.array([1.5708, -1.3, 2.1, -0.80, 1.5708, 3.14159])
initial_qpos_simpheg = numpy.array([1.5708, -1.3, 2.1, -0.80, 1.5708, 0])
initial_qpos_simpheg_conf2 = numpy.array([0, -1.3, 2.1, -0.80, 1.5708, 0])

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

class Ur10HegRelEnv(ur10_corrective_env.Ur10Env, utils.EzPickle):
    def __init__(self, reward_type='dense'):
        ur10_corrective_env.Ur10Env.__init__(
            self, MODEL_XML_PATH_SLOW_SH_CONF2, n_substeps=100, distance_threshold=0.001,
            initial_qpos=initial_qpos_simpheg_conf2, reward_type=reward_type, ctrl_type='cartesian', corrective=False)
        utils.EzPickle.__init__(self)

class Ur10HegRelVaryEnv(ur10_corrective_env.Ur10Env, utils.EzPickle):
    def __init__(self, reward_type='dense'):
        ur10_corrective_env.Ur10Env.__init__(
            self, MODEL_XML_PATH_SLOW_SH_CONF2, n_substeps=100, distance_threshold=0.001,
            initial_qpos=initial_qpos_simpheg_conf2, reward_type=reward_type, ctrl_type='cartesian', corrective=False, vary=True)
        utils.EzPickle.__init__(self)

class Ur10HegCorrectiveEnv(ur10_corrective_env.Ur10Env, utils.EzPickle):
    def __init__(self, reward_type='dense'):
        ur10_corrective_env.Ur10Env.__init__(
            self, MODEL_XML_PATH_SLOW_SH_CONF2, n_substeps=100, distance_threshold=0.001,
            initial_qpos=initial_qpos_simpheg_conf2, reward_type=reward_type, ctrl_type='cartesian')
        utils.EzPickle.__init__(self)

class Ur10HegCorrectiveVaryEnv(ur10_corrective_env.Ur10Env, utils.EzPickle):
    def __init__(self, reward_type='dense'):
        ur10_corrective_env.Ur10Env.__init__(
            self, MODEL_XML_PATH_SLOW_SH_CONF2, n_substeps=100, distance_threshold=0.001,
            initial_qpos=initial_qpos_simpheg_conf2, reward_type=reward_type, ctrl_type='cartesian', vary=True)
        utils.EzPickle.__init__(self)

class Ur10HegRandEnv(ur10_noisy_pd_env.Ur10Env, utils.EzPickle):
    def __init__(self, reward_type='dense'):
        ur10_noisy_pd_env.Ur10Env.__init__(
            self, MODEL_XML_PATH_RAND, n_substeps=80, distance_threshold=0.001,
            initial_qpos=initial_qpos_simpheg_conf2, reward_type=reward_type, ctrl_type='cartesian', vary=False)
        utils.EzPickle.__init__(self)