import os
import numpy
from gym import utils
from gym.envs.robotics import ur10_env
from gym.envs.robotics import ur10_rel_env
from gym.envs.robotics import ur10_rel_simpheg_env
from gym.envs.robotics import ur10_corrective_env
from gym.envs.robotics import ur10_noisy_pd_env
from gym.envs.robotics import ur10_static_pd_env
from gym.envs.robotics import ur10_static_env
from gym.envs.robotics import ur10_noisy_position_env
from gym.envs.robotics import ur10_noisy_force_env
from gym.envs.robotics import ur10_rand_force_env

# Ensure we get the path separator correct on windows
PROJECT_PATH = os.path.join(*[os.getenv("HOME"), "DRL_AI4RoMoCo"])
MODEL_PATH = os.path.join(*[PROJECT_PATH, "code", "environment" ,"UR10"])
#MODEL_XML_PATH = os.path.join(*[MODEL_PATH,"ur10_heg.xml"])
#MODEL_XML_PATH_SLOW = os.path.join(*[MODEL_PATH,"ur10_heg_slow.xml"])
#MODEL_XML_PATH_SLOW_SH = os.path.join(*[MODEL_PATH, "ur10_heg_slow_simpheg.xml"])
#MODEL_XML_PATH_SLOW_SH_CONF2 = os.path.join(*[MODEL_PATH, "ur10_heg_slow_simpheg_conf2.xml"])
#MODEL_XML_PATH_RAND = os.path.join(*[MODEL_PATH, "ur10_assembly_setup_rand_temp_1.xml"])
#MODEL_XML_PATH_STATIC = os.path.join(*[MODEL_PATH, "ur10_heg_static.xml"])
#MODEL_XML_PATH_STATIC_PD = os.path.join(*[MODEL_PATH, "ur10_heg_static_pd.xml"])
#MODEL_XML_PATH_STATIC = os.path.join(*[PROJECT_PATH, "code", "environment", "UR10_new", "ur10_heg.xml"])

static_position_config_file = "env_config_002.yml"
noisy_position_config_file = "env_config_003.yml"
static_force_config_file = "env_config_000.yml"
noisy_force_config_file="env_config_004.yml"
rand_force_config_file = "env_config_001.yml"
STATIC_POSITION_CONFIG_PATH = os.path.join(*[PROJECT_PATH,"code", "configs", "environment", static_position_config_file])
NOISY_POSITION_CONFIG_PATH = os.path.join(*[PROJECT_PATH,"code", "configs", "environment", noisy_position_config_file])
STATIC_FORCE_CONFIG_PATH = os.path.join(*[PROJECT_PATH,"code", "configs", "environment", static_force_config_file])
NOISY_FORCE_CONFIG_PATH = os.path.join(*[PROJECT_PATH,"code", "configs", "environment", noisy_force_config_file])
RAND_FORCE_CONFIG_PATH = os.path.join(*[PROJECT_PATH,"code", "configs", "environment", rand_force_config_file])

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
'''
class Ur10HegRandEnv(ur10_noisy_pd_env.Ur10Env, utils.EzPickle):
    def __init__(self, reward_type='dense'):
        ur10_noisy_pd_env.Ur10Env.__init__(
            self, MODEL_XML_PATH_RAND, n_substeps=80, distance_threshold=0.005,
            initial_qpos=initial_qpos_simpheg_conf2, reward_type=reward_type, ctrl_type='cartesian', vary=False)
        utils.EzPickle.__init__(self)

class Ur10HegStaticEnv(ur10_static_pd_env.Ur10Env, utils.EzPickle):
    def __init__(self, reward_type='dense'):
        ur10_static_pd_env.Ur10Env.__init__(
            self, MODEL_XML_PATH_STATIC_PD, n_substeps=80, distance_threshold=0.005,
            initial_qpos=initial_qpos_simpheg_conf2, reward_type=reward_type, ctrl_type='cartesian')
        utils.EzPickle.__init__(self)

class Ur10HegGenesisEnv(ur10_static_env.Ur10Env, utils.EzPickle):
    def __init__(self, reward_type='dense'):
        ur10_static_env.Ur10Env.__init__(
            self, MODEL_XML_PATH_STATIC, n_substeps=80, distance_threshold=0.005,
            initial_qpos=initial_qpos_simpheg_conf2, reward_type=reward_type, ctrl_type='cartesian')
        utils.EzPickle.__init__(self)
'''
################################################
############ Using config files ################
################################################

# UR10HEG-v001
class Ur10HegRandEnv(ur10_noisy_pd_env.Ur10Env, utils.EzPickle):
    def __init__(self):
        ur10_noisy_pd_env.Ur10Env.__init__(self, RAND_FORCE_CONFIG_PATH)
        utils.EzPickle.__init__(self)

class Ur10HegRandForceEnv(ur10_rand_force_env.Ur10Env, utils.EzPickle):
    def __init__(self):
        ur10_rand_force_env.Ur10Env.__init__(
            self, 
            RAND_FORCE_CONFIG_PATH,
            model_xml_path = os.path.join(*[PROJECT_PATH, "code", "environment", "UR10_Force_Randomized", "ur10_assembly_setup_rand_temp_0.xml"]),
            worker_id = 1,
            )
        utils.EzPickle.__init__(self)

#UR10HEG-v000
class Ur10HegStaticEnv(ur10_static_pd_env.Ur10Env, utils.EzPickle):
    def __init__(self):
        ur10_static_pd_env.Ur10Env.__init__(self, STATIC_FORCE_CONFIG_PATH)
        utils.EzPickle.__init__(self)
#UR10HEG-v002
class Ur10HegGenesisEnv(ur10_static_env.Ur10Env, utils.EzPickle):
    def __init__(self):
        ur10_static_env.Ur10Env.__init__(self, STATIC_POSITION_CONFIG_PATH)
        utils.EzPickle.__init__(self)

#UR10HEG-v003
class Ur10HegNoisyPositionEnv(ur10_noisy_position_env.Ur10Env, utils.EzPickle):
    def __init__(self):
        ur10_noisy_position_env.Ur10Env.__init__(self, NOISY_POSITION_CONFIG_PATH)
        utils.EzPickle.__init__(self)

#UR10HEG-v004
class Ur10HegNoisyForceEnv(ur10_noisy_force_env.Ur10Env, utils.EzPickle):
    def __init__(self):
        ur10_noisy_force_env.Ur10Env.__init__(self, NOISY_FORCE_CONFIG_PATH)
        utils.EzPickle.__init__(self)