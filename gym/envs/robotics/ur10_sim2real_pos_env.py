import numpy as np
import numpy
import yaml
import mujoco_py
import json
from gym.envs.robotics import rotations, robot_custom_env_mh,robot_custom_env, utils
import os
from gym.envs.robotics.ur10 import randomize
import matplotlib.pyplot as plt
from scipy.signal import lfilter, lfilter_zi, butter

PROJECT_PATH = os.path.join(*[os.getenv("HOME"), "DRL_AI4RoMoCo"])
SAVE_PATH = os.path.join(*[
    PROJECT_PATH, 
    "code", "data", 
    "EVAL_TARGETSIM",
    "NoisyVaryEnv_Actions"
    ])
GOAL_PATH = os.path.join(*[
    PROJECT_PATH,
    "code",
    "environment",
    "experiment_configs",
    "goal_ur10_simpheg_conf2.json"
])

def goal_distance(obs, goal):
    obs = obs[:6]
    assert obs.shape == goal.shape
    return np.linalg.norm(obs*np.array([1, 1, 1, 0.3, 0.3, 0.3]), axis=-1)

def normalize_rad(angles):
    angles = np.array(angles)
    angles = angles % (2*np.pi)
    angles = (angles + 2*np.pi) % (2*np.pi)
    for i in range(len(angles)):
        if (angles[i] > np.pi):
            angles[i] -= 2*np.pi
    return angles


class Ur10Env(robot_custom_env.RobotEnv):
    """Superclass for all Ur10 environments.
    """

    def __init__(self,env_config, model_xml_path, worker_id):

        with open(env_config) as cfg:
            env_config = yaml.load(cfg, Loader=yaml.FullLoader)

        if model_xml_path == None:
            self.model_path = env_config["model_xml_file"]

    ########################################## MONITORING

        self.episode = 0
        self.results = list()

        self.start_flag = True
        self.save_data = env_config["Saving"]
        if self.save_data:
            self.fts = []
            self.fxs = []
            self.fys = []
            self.fzs = []
            self.obs = []
            self.rewards = []
            self.poses = []

    ##########################################
        self.model_path = model_xml_path
        self.n_substeps = env_config["n_substeps"]
        self.distance_threshold = env_config["Learning"]["distance_threshold"]
        self.fail_threshold = env_config["Learning"]["fail_threshold"]
        self.initial_qpos = numpy.array(env_config["initial_qpos"])
        self.reward_type = env_config["reward_type"]
        self.ctrl_type = env_config["ctrl_type"]
        #self.corrective = env_config["corrective"]

        self.n_actions = env_config["n_actions"]
        self.action_rate = env_config["action_rate"] 
        self.SEED = env_config["SEED"]

        self.R1 = env_config["Reward"]["R1"]
        self.R2 = env_config["Reward"]["R2"]
        self.success_reward = env_config["Reward"]["success_reward"]

        self.dx_max = env_config["dx_max"]
        self.gripper_mass = env_config["gripper_mass"]

    ########################################## NOISE

        self.pos_noise_std = numpy.array(env_config["Noise"]["pos_noise_std"])
        self.ft_noise_std = numpy.array(env_config["Noise"]["ft_noise_std"])
        self.dq_noise_std = numpy.array(env_config["Noise"]["dq_noise_std"])

    ########################################## RANDOMIZATION
    
        self.vary = env_config["Domain_Randomization"]["vary"]
        self.vary = env_config["Domain_Randomization"]["vary"]
        self.vary_params = env_config["Domain_Randomization"]["vary_params"]    
        self.init_vary_range = numpy.concatenate([                              
            numpy.array(self.vary_params[:3]),
            numpy.array(self.vary_params[3:])*(numpy.pi/180)
            ])
        self.pos_rand_uncor_bound = numpy.array(
            env_config["Domain_Randomization"]["pos_rand_uncor_bound"])
        self.ft_rand_uncor_bound = numpy.array(
            env_config["Domain_Randomization"]["ft_rand_uncor_bound"])
        self.pos_rand_cor_bound = numpy.array(
            env_config["Domain_Randomization"]["pos_rand_cor_bound"])
        self.ft_rand_cor_bound = numpy.array(
            env_config["Domain_Randomization"]["ft_rand_cor_bound"])
        self.pos_rand_cor = numpy.array(
            env_config["Domain_Randomization"]["pos_rand_cor"])
        self.ft_rand_cor = numpy.array(
            env_config["Domain_Randomization"]["ft_rand_cor"])
        self.randomize_kwargs = env_config["randomize_kwargs"]
        self.offset = randomize.randomize_ur10_xml()
        self.worker_id = worker_id

    ######################################## CONTROLLER PARAMETERS

        #self.K = numpy.array(env_config["controller"]["K"])
        #self.kp = numpy.array(env_config["controller"]["kp"])
        #self.ki = numpy.array(env_config["controller"]["ki"])
        #self.kd = numpy.array(env_config["controller"]["kd"])
        #self.kf = numpy.array(env_config["controller"]["kf"])
        #self.max_fb = numpy.array(env_config["controller"]["max_fb"])
        #self.ctrl_buffer = numpy.repeat(
        #    self.initial_qpos.copy().reshape(1, 6), 4, axis=0)
        #b, a = butter(
        #    env_config["controller"]["butter_0"], 
        #    env_config["controller"]["butter_1"])
        #self.a = a
        #self.b = b
        #self.zi = [lfilter_zi(b,a) * self.initial_qpos[i] for i in range(6)]
        #self.qi_diff = env_config["controller"]["qi_diff"]
        #self.last_target_q = self.initial_qpos.copy()
        #self.only_grav_comp = env_config["controller"]["only_grav_comp"]
        #self.sim_ctrl_q = self.initial_qpos

    #########################################

        super(Ur10Env, self).__init__(
            model_path=self.model_path, n_substeps=self.n_substeps, 
            n_actions=self.n_actions, initial_qpos=self.initial_qpos, 
            seed=self.SEED, success_reward=self.success_reward, 
            action_rate=self.action_rate)
        
    def activate_noise(self):
        self.vary=True
        print('noise has been activated.')

    def compute_reward(self, obs, goal, info):
        d = goal_distance(obs,goal)
        f = numpy.absolute(obs[7])
        + numpy.absolute(obs[8])
        + numpy.absolute(obs[9])
        rew = self.R1 * (-d) + self.R2 *(-f)
        if self.save_data:
            self.rewards.append(rew)
        self.step_count += 1
        return rew

    def _step_callback(self):
        a = 0

    def set_state(self, qpos):
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, old_state.qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def _set_action(self, action):

        assert action.shape == (6,)
        action = action.copy()  # ensure that we don't change the action outside of this scope

        deviation = sum(abs(self.sim.data.qpos - self.sim.data.ctrl))
        # print(deviation )
        if  deviation > 0.35:  # reset control to current position if deviation too high
            self.sim.data.ctrl[:] = self.sim.data.qpos 
            + self.get_dq([0, 0, 0.005, 0, 0, 0])
            #print('deviation compensated')

        if self.ctrl_type == "joint":
            action *= 0.05  # limit maximum change in position
            # Apply action #scalarsto simulation.
            utils.ctrl_set_action(self.sim, action)
        elif self.ctrl_type == "cartesian":
            dx = action.reshape(6, )

            max_limit = self.dx_max
            # limitation of operation space, we only allow small rotations adjustments in x and z directions, moving in y direction
            x_now = numpy.concatenate((
                self.sim.data.get_body_xpos("gripper_dummy_heg"), 
                self.sim.data.get_body_xquat("gripper_dummy_heg")))
            x_then = x_now[:3] + dx[:3]*max_limit

            #diff_now = numpy.array(x_now - self.init_x).reshape(7,)
            diff_then = numpy.array(x_then[:3] - self.init_x[:3])

            barriers_min = numpy.array([-0.4, -0.8,   -0.4])
            barriers_max = numpy.array([0.4,  0.8, 0.4])

            #for i in range(3):
            #    if (barriers_min[i] < diff_then[i] < barriers_max[i]):
            #    dx[i] = dx[i] * max_limit
            #    elif barriers_min[i] > diff_then[i]:
            #        dx[i] = + max_limit
            #    elif barriers_max[i] < diff_then[i]:
            #        dx[i] = - max_limit
            for i in range(6):
                dx[i] = dx[i] * max_limit

            #rot_mat = self.sim.data.get_body_xmat('gripper_dummy_heg')
            #dx_ = np.concatenate([rot_mat.dot(dx[:3]), rot_mat.dot(dx[3:])])
            #dx_[2]+= 1
            dq = self.get_dq(dx)
            for i in range(6):
                self.sim.data.ctrl[i] += dq[i]


    def get_dq(self, dx):
        jacp = self.sim.data.get_body_jacp(name="gripper_dummy_heg").reshape(3, 6)
        jacr = self.sim.data.get_body_jacr(name="gripper_dummy_heg").reshape(3, 6)
        jac = numpy.vstack((jacp, jacr))
        dq = numpy.linalg.lstsq(jac, dx)[0].reshape(6, )
        return dq


    def _get_obs(self):
        # positions
        rot_mat = self.sim.data.get_body_xmat('gripper_dummy_heg')
        ft = self.sim.data.sensordata.copy()

        if self.start_flag:
            ft = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.start_flag = False

        ft += numpy.random.normal(0.0, self.ft_noise_std)   
        ft += numpy.random.uniform(-self.ft_rand_uncor_bound, self.ft_rand_uncor_bound) 
        ft += self.ft_rand_cor

        ft[1] -= self.gripper_mass*9.81
        x_pos = self.sim.data.get_body_xpos("gripper_dummy_heg")
        x_pos += numpy.random.normal(0.0, self.pos_noise_std[:3])
        x_pos += numpy.random.uniform(
            -self.pos_rand_uncor_bound[:3], 
            self.pos_rand_uncor_bound[:3])
        x_pos += self.pos_rand_cor[:3]

        #x_quat = self.sim.data.get_body_xquat("gripper_dummy_heg")
        x_mat = self.sim.data.get_body_xmat("gripper_dummy_heg")
        rpy = normalize_rad(rotations.mat2euler(x_mat)
                            + numpy.random.normal(
                                0.0, 
                                self.pos_noise_std[3:]) * (numpy.pi/180)
                            + numpy.random.uniform(
                                -self.pos_rand_uncor_bound[3:],
                                self.pos_rand_uncor_bound[3:]) * (numpy.pi/180)
                            + self.pos_rand_cor[3:] * (numpy.pi/180)) 

        obs = np.concatenate([
            rot_mat.dot(
                x_pos-self.goal[:3]), 
                rot_mat.dot(normalize_rad(rpy-self.goal[3:])), 
                ft.copy()
        ])

        if self.save_data:
            self.fts.append([ft[0], ft[1], ft[2], ft[3], ft[4], ft[5],])
            self.obs.append(obs)
            self.fxs.append(ft[0])
            self.fys.append(ft[1])
            self.fzs.append(ft[2])
            self.poses.append(numpy.concatenate(
                [x_pos-self.goal[:3],
                normalize_rad(rpy-self.goal[3:])]))

        self.last_obs = obs
        return obs

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('body_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        pass

    def _reset_sim(self):

        self.start_flag = True
        if self.episode > 0:
            self.success_rate = float(numpy.sum(self.results)/float(len(self.results)))
            print(" | Episode: {} | Success Rate: {} | ".format(
                self.episode, 
                self.success_rate)
                )
        if len(self.results) < 10:
            self.results.append(0)
        else:
            self.results.pop(0)
            self.results.append(0)

        if self.save_data and self.episode > 0:
            if self.success_flag == 1:
                self.rewards.append(self.success_reward)
            self.reward_sum = numpy.sum(self.rewards)
            
            save_dict = {
                    "fx" : self.fxs,
                    "fy" : self.fys,
                    "fz" : self.fzs,
                    "steps" : self.step_count,
                    "success" : self.success_flag,
                    "reward" : self.reward_sum
            }
            with open(os.path.join(*[
                SAVE_PATH, "episode_{}.json".format(self.episode)]), "w") as file:
                json.dump(save_dict,file)
                file.write('\n')

        self.fxs = []
        self.fys = []
        self.fzs = []
        self.rewards = []
        self.step_count = 0
        self.success_flag = 0
        self.episode += 1

        if self.vary:
            deviation_x = numpy.random.uniform(
                -self.init_vary_range, self.init_vary_range)
            deviation_q = self.get_dq(deviation_x)
        else:
            deviation_q = numpy.array([0,0,0,0,0,0])

        self.pos_rand_cor = numpy.random.uniform(
            -self.pos_rand_cor_bound, self.pos_rand_cor_bound)
        self.ft_rand_cor = numpy.random.uniform(
            -self.ft_rand_cor_bound, self.ft_rand_cor_bound)
        self.pos_rand_cor = numpy.random.uniform(
            -self.pos_rand_cor_bound, self.pos_rand_cor_bound)

        del self.sim
        self.offset = randomize.randomize_ur10_xml(
            worker_id=self.worker_id, **self.randomize_kwargs)
        model = mujoco_py.load_model_from_path(self.model_path)
        self.sim = mujoco_py.MjSim(model, nsubsteps=self.n_substeps)
        self.goal = self._sample_goal()
        if not self.viewer is None:
            self._get_viewer('human').update_sim(self.sim)

        #self.ctrl_buffer = np.repeat((
        #    self.initial_qpos + deviation_q).reshape(1, 6), 4, axis=0)
        #self.b, self.a = butter(2, 0.12)
        #self.zi = [lfilter_zi(self.b, self.a) * (self.initial_qpos[i] + deviation_q[i]) for i in range(6)]
        #self.qi_diff = 0
        #self.last_target_q = self.initial_qpos + deviation_q

        self.set_state(self.initial_qpos + deviation_q)
        self.sim.forward()

        #self.sim.step()

        self.init_x = numpy.concatenate(
            (self.sim.data.get_body_xpos("gripper_dummy_heg"), 
            self.sim.data.get_body_xquat("gripper_dummy_heg")
            ))
        self.sim.data.ctrl[:] = self.initial_qpos + deviation_q

        return True

    def _sample_goal(self):
        home_path = os.getenv("HOME")
        goal_path = os.path.join(*[
            home_path, "DRL_AI4RoMoCo", "code", "environment", 
            "experiment_configs", "goal_ur10_simpheg_conf2.json"])

        with open(goal_path, encoding='utf-8') as file:
            goal = json.load(file)
            xpos =  goal['xpos'] + self.offset['body_pos_offset']
            xquat = goal['xquat']
            rpy = normalize_rad(rotations.quat2euler(xquat)+self.offset['body_euler_offset'])
        return numpy.concatenate([xpos, rpy]).copy()

    def _is_success(self, achieved_goal, desired_goal):
        rot_mat = self.sim.data.get_body_xmat('gripper_dummy_heg')
        x_pos = self.sim.data.get_body_xpos("gripper_dummy_heg")
        x_mat = self.sim.data.get_body_xmat("gripper_dummy_heg")
        rpy = normalize_rad(rotations.mat2euler(x_mat))
        obs = np.concatenate([
            rot_mat.dot(x_pos-self.goal[:3]), rot_mat.dot(normalize_rad(rpy-self.goal[3:]))
        ])

        d = goal_distance(obs, desired_goal)

        if d < self.distance_threshold:
            if len(self.results) == 0:
                self.results.append(1)
            else:
                self.results.pop()
                self.results.append(1)
            self.success_flag = 1
            return True
        else:
            return False 

    def _is_failure(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return False

    def _env_setup(self, initial_qpos):
        self.sim.data.ctrl[:] = initial_qpos
        self.set_state(initial_qpos)
        self.sim.forward()


    def render(self, mode='human', width=500, height=500):
        return super(Ur10Env, self).render(mode, width, height)
