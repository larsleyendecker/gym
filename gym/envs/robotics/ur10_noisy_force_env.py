import os
import numpy
import yaml
import pandas
import mujoco_py
import json
from gym.envs.robotics import rotations, robot_custom_env, utils
from gym.envs.robotics.ur10 import randomize
from scipy.signal import lfilter, lfilter_zi, butter
#from utils.saving import NumpyEncoder

PROJECT_PATH = os.path.join(*[os.getenv("HOME"), "DRL_AI4RoMoCo"])
MODEL_PATH = os.path.join(*[PROJECT_PATH, "code", "environment", "UR10_Force"])
CONFIG_PATH = os.path.join(*[PROJECT_PATH, "code", "config", "environment"])
SAVE_PATH = os.path.join(*[
    PROJECT_PATH,
    "code",
    "data",
    "TEST_SIM",
    "NoisyForceEnv",
    "Model4"
    ])

GOAL_PATH = os.path.join(*[
    PROJECT_PATH,
    "code",
    "environment",
    "experiment_configs",
    "goal_ur10_simpheg_conf2.json"
])

def goal_distance(obs, goal):
    '''Computation of the distance between gripper and goal'''
    obs = obs[:6]
    assert obs.shape == goal.shape
    return numpy.linalg.norm(obs*numpy.array([1, 1, 1, 0.3, 0.3, 0.3]), axis=-1)

def normalize_rad(angles):
    '''Normalizing Euler angles'''
    angles = numpy.array(angles)
    angles = angles % (2*numpy.pi)
    angles = (angles + 2*numpy.pi) % (2*numpy.pi)
    for i in range(len(angles)):
        if (angles[i] > numpy.pi):
            angles[i] -= 2*numpy.pi
    return angles

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Ur10Env(robot_custom_env.RobotEnv):
    """Superclass for all Ur10 environments."""

    def __init__(self, env_config):

        with open(env_config) as cfg:
            env_config = yaml.load(cfg, Loader=yaml.FullLoader)

        self.save_data = env_config["Saving"]
        self.start_flag = True
        self.episode = 0
        if self.save_data:
            self.fts = []
            self.fxs = []
            self.fys = []
            self.fzs = []
            self.obs = []
            self.rewards = []
            self.poses = []
        self.SEED = env_config["SEED"]
        self.run_info = env_config["info"]
        self.results = list(numpy.zeros(10,).astype(int))
        self.R1 = env_config["Reward"]["R1"]
        self.R2 = env_config["Reward"]["R2"]
        self.success_reward = env_config["Reward"]["success_reward"]
        self.model_path = os.path.join(*[MODEL_PATH,env_config["model_xml_file"]])
        self.initial_qpos = numpy.array(env_config["initial_qpos"])           
        self.sim_ctrl_q = self.initial_qpos                                         
        self.reward_type = env_config["reward_type"]                                
        self.ctrl_type = env_config["ctrl_type"]                            
        self.n_substeps = env_config["n_substeps"]      
        self.action_rate = env_config["action_rate"]                           
        self.distance_threshold = env_config["Learning"]["distance_threshold"]
        self.cur_eps_threshold = env_config["Learning"]["cur_eps_threshold"]
        self.curriculum_learning = env_config["Learning"]["curriculum_learning"]
        self.initial_distance_threshold = env_config["Learning"]["initial_distance_threshold"]
        self.final_distance_threshold = env_config["Learning"]["final_distance_threshold"]
        self.fail_threshold = env_config["Learning"]["fail_threshold"]                          
        self.n_actions = env_config["n_actions"]
        self.corrective = env_config["corrective"]
        self.vary = env_config["vary"]
        self.dx_max = env_config["dx_max"]
        self.only_grav_comp = True

        # Controller Parameter
        self.K = numpy.array(env_config["controller"]["K"])
        self.kp = numpy.array(env_config["controller"]["kp"])
        self.ki = numpy.array(env_config["controller"]["ki"])
        self.kd = numpy.array(env_config["controller"]["kd"])
        self.kf = numpy.array(env_config["controller"]["kf"])
        self.max_fb = numpy.array(env_config["controller"]["max_fb"])

        self.ctrl_buffer = numpy.repeat(self.initial_qpos.copy().reshape(1, 6), 4, axis=0)
        b, a = butter(env_config["controller"]["butter_0"], env_config["controller"]["butter_1"])
        self.a = a
        self.b = b
        self.zi = [lfilter_zi(b,a) * self.initial_qpos[i] for i in range(6)]
        self.qi_diff = env_config["controller"]["qi_diff"]
        self.last_target_q = self.initial_qpos.copy()

        self.f_mean = env_config["Noise"]["f_mean"]
        self.t_mean = env_config["Noise"]["t_mean"]
        self.pos_mean = env_config["Noise"]["pos_mean"]
        self.rot_mean = env_config["Noise"]["rot_mean"]
        self.f_std_si = env_config["Noise"]["f_std_si"]
        self.t_std_si = env_config["Noise"]["t_std_si"]
        self.pos_std_si = env_config["Noise"]["pos_std_si"]
        self.rot_std_si = env_config["Noise"]["rot_std_si"]
        self.f_std_dr = env_config["Noise"]["f_std_dr"]
        self.t_std_dr = env_config["Noise"]["t_std_dr"]
        self.pos_std_dr = env_config["Noise"]["pos_std_dr"]
        self.rot_std_dr = env_config["Noise"]["rot_std_dr"]

        ############################
        
        super(Ur10Env, self).__init__(
            model_path=self.model_path, n_substeps=self.n_substeps,
            n_actions=self.n_actions, initial_qpos=self.initial_qpos, 
            seed=self.SEED, success_reward=self.success_reward,
            action_rate=self.action_rate)

    def set_force_for_q(self, q_ctrl, only_grav_comp=False):
        '''
           Takes positional joint space ctrl signal from set_action and coverts it
           into forces for force control (Not required for position control)
        '''
        q_ctrl = q_ctrl.reshape(6, )

        self.ctrl_buffer[1:] = self.ctrl_buffer[:3].copy()
        self.ctrl_buffer[0] = q_ctrl.copy()
        target_q = numpy.zeros(6)

        for i in range(6):
            target_q[i], zo = lfilter(self.b, self.a, [self.ctrl_buffer[3][i]], zi=self.zi[i])
            self.zi[i] = zo.copy()

        target_qd = (target_q - self.last_target_q).copy() * 125
        self.last_target_q = target_q.copy()

        q_diff = target_q - self.sim.data.qpos
        qd_diff = target_qd - self.sim.data.qvel
        self.qi_diff += q_diff

        #for j in range(6):
        #    if numpy.sign(self.qi_diff[j]) != numpy.sign(q_diff[j]):
        #        self.qi_diff[j] = 0

        self.sim.data.ctrl[:] = self.sim.data.qfrc_bias.copy()

        if not only_grav_comp:
            self.sim.data.ctrl[:] += self.K * (numpy.clip(self.kp * q_diff + self.ki * self.qi_diff,
                                                       -self.max_fb, self.max_fb) + self.kf * target_qd)
        self.sim_ctrl_q = q_ctrl
        return self.sim.data.ctrl[:].copy()

    def compute_reward(self, obs, goal, info):
        d = goal_distance(obs,goal)
        f = numpy.absolute(obs[7])
        + numpy.absolute(obs[8])
        + numpy.absolute(obs[9])
        rew = self.R1 * (-d) + self.R2 *(-f)
        if self.save_data:
            self.rewards.append(rew)
        return rew

    def _step_callback(self):
        pass
        
    def set_state(self, qpos):
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(
            old_state.time, 
            qpos, 
            old_state.qvel, 
            old_state.act, 
            old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def _set_action(self, action):
        '''Interface to the DRL agent: the agent outputs a positional control signal
        based on the current position q. _set_action takes this positional action
        and  
        '''
        assert action.shape == (6,)
        # ensure that we don't change the action outside of this scope
        action = action.copy()
        deviation = sum(abs(self.sim.data.qpos - self.sim_ctrl_q))
        
        # reset control to current position if deviation too high
        if  deviation > 0.35:
            self.set_force_for_q(self.sim.data.qpos)
            print('deviation compensated')

        if self.ctrl_type == "joint":
            action *= 0.05  # limit maximum change in position
            # Apply action #scalarsto simulation.
            utils.ctrl_set_action(self.sim, action)
        elif self.ctrl_type == "cartesian":
            dx = action.reshape(6, )

            max_limit = self.dx_max
            '''
            # limitation of operation space, 
            # we only allow small rotations adjustments in x and z directions, 
            # moving in y direction
            '''
            x_now = numpy.concatenate(
                (self.sim.data.get_body_xpos("gripper_dummy_heg"),
                self.sim.data.get_body_xquat("gripper_dummy_heg"))
                )
            x_then = x_now[:3] + dx[:3]*max_limit

            #diff_now = numpy.array(x_now - self.init_x).reshape(7,)
            diff_then = numpy.array(x_then[:3] - self.init_x[:3])
            
            #Default
            #barriers_min = numpy.array([-0.2, -0.2,   -0.2])
            #barriers_max = numpy.array([0.2,  0.4, 0.2])
            #These barriers do the operation space limitation 
            # [x_range, y_range, z_range]
            barriers_min = numpy.array([-0.4, -0.8,   -0.4])
            barriers_max = numpy.array([0.4,  0.8, 0.4])



            for i in range(6):
                dx[i] = dx[i] * max_limit

            if self.corrective:
                # bias in direction of assembly
                bias_dir = -self.last_obs[:6]
                # print(bias_dir)
                for i in range(3,6):
                    if bias_dir[i] > 0.5:
                        print(i, bias_dir[i])
                    bias_dir[i] = bias_dir[i] # slower rotations
                bias_dir /= numpy.linalg.norm(bias_dir)
                # print(bias_dir)
                dx += bias_dir * max_limit * 0.5
                dx.reshape(6, 1)

            rot_mat = self.sim.data.get_body_xmat('gripper_dummy_heg')
            ## transform to right coordinate system
            dx_ = numpy.concatenate([rot_mat.dot(dx[:3]), rot_mat.dot(dx[3:])]) 
            #dx_[2]+= 1
            dq = self.get_dq(dx_)
            q = self.sim_ctrl_q + dq
            self.set_force_for_q(q)
            # print(sum(abs(sim.data.qpos-sim.data.ctrl)))

    def get_dq(self, dx):
        '''Does the transformation from cartesian space (dx) to joint space (dq)
           Required since actions from the agent are in cartesian space, but the 
           robot is controlled in joint space for each of the six joint angles
        '''
        jacp = self.sim.data.get_body_jacp(name="gripper_dummy_heg").reshape(3, 6)
        jacr = self.sim.data.get_body_jacr(name="gripper_dummy_heg").reshape(3, 6)
        jac = numpy.vstack((jacp, jacr))
        dq = numpy.linalg.lstsq(jac, dx)[0].reshape(6, )
        return dq

    def _get_obs(self):
        rot_mat = self.sim.data.get_body_xmat('gripper_dummy_heg')
        ft = self.sim.data.sensordata.copy()
        ft[:3] += numpy.random.normal(self.f_mean, (self.f_std_si + self.f_std_dr), 3)
        ft[3:] += numpy.random.normal(self.t_mean, (self.t_std_si + self.t_std_dr), 3)

        if self.start_flag:
            ft = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            ft[:3] += numpy.random.normal(self.f_mean, (self.f_std_si + self.f_std_dr), 3)
            ft[3:] += numpy.random.normal(self.t_mean, (self.t_std_si + self.t_std_dr), 3)
        self.start_flag = False

        x_pos = self.sim.data.get_body_xpos("gripper_dummy_heg")
        x_pos += numpy.random.normal(self.pos_mean, (self.pos_std_si + self.pos_std_dr), 3)
        x_mat = self.sim.data.get_body_xmat("gripper_dummy_heg")
        rpy = normalize_rad(
            rotations.mat2euler(x_mat)
            + numpy.random.normal(self.rot_mean, (self.rot_std_si + self.rot_std_dr), 3)
            )

        obs = numpy.concatenate([
            rot_mat.dot(x_pos-self.goal[:3]),
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
        # Tracking the first step to zero the ft-sensor
        self.start_flag = True
        self.success_rate = numpy.sum(self.results)/len(self.results)
        self.results.pop(0)
        self.results.append(0)
        print("Episode: {} Success Rate: {} ".format(self.episode, self.success_rate))
        
        if self.save_data and self.episode > 0:
            
            save_dict = {
                    #"observations" : self.obs,
                    #"ft_values" : self.fts,
                    #"rewards" : self.rewards,
                    #"poses" : self.poses
                    "fx" : self.fxs,
                    "fy" : self.fys,
                    "fz" : self.fzs
            }
            with open(os.path.join(*[SAVE_PATH, "episode_{}.json".format(self.episode)]), "w") as file:
                json.dump(save_dict,file)
                file.write('\n')
        
            self.obs = []
            self.fts = []
            self.rewards = []
            self.poses = []
            self.fxs = []
            self.fys = []
            self.fzs = []
        self.episode += 1
        
        #if not self.viewer is None:
        #    self._get_viewer('human').update_sim(self.sim)
        #self._get_viewer('human')._ncam = self.sim.model.ncam
        
        deviation_q = numpy.array([0, 0, 0, 0, 0, 0])

        self.ctrl_buffer = numpy.repeat(
            (self.initial_qpos + deviation_q).reshape(1, 6),
            4,
            axis=0)
        self.b, self.a = butter(2, 0.12)
        self.zi = [lfilter_zi(self.b, self.a) * (self.initial_qpos[i] + deviation_q[i]) for i in range(6)]
        self.qi_diff = 0
        self.last_target_q = self.initial_qpos + deviation_q
        
        self.set_state(self.initial_qpos + deviation_q)
        self.sim.forward()
        self.sim.step()
        
        # Checking if there are contacts in simulation and returning false if so
        if self.sim.data.ncon == 0:
            for i in range(100):
                self.sim.data.ctrl[:] = self.sim.data.qfrc_bias.copy()
                self.sim.forward()
                self.sim.step()
            self.init_x = numpy.concatenate(
                (self.sim.data.get_body_xpos("gripper_dummy_heg"),
                self.sim.data.get_body_xquat("gripper_dummy_heg"))
                )
            self.set_force_for_q(self.initial_qpos + deviation_q)

        return self.sim.data.ncon == 0

    def _sample_goal(self):

        with open(GOAL_PATH, encoding='utf-8') as file:
            goal = json.load(file)
            xpos =  goal['xpos']
            xquat = goal['xquat']
            rpy = normalize_rad(rotations.quat2euler(xquat))
        return numpy.concatenate([xpos, rpy]).copy()

    def _is_success(self, achieved_goal, desired_goal):
        rot_mat = self.sim.data.get_body_xmat('gripper_dummy_heg')
        x_pos = self.sim.data.get_body_xpos("gripper_dummy_heg")
        x_mat = self.sim.data.get_body_xmat("gripper_dummy_heg")
        rpy = normalize_rad(rotations.mat2euler(x_mat))
        obs = numpy.concatenate([
            rot_mat.dot(x_pos-self.goal[:3]), 
            rot_mat.dot(normalize_rad(rpy-self.goal[3:]))
        ])
        d = goal_distance(obs, desired_goal)
        if self.curriculum_learning:
            if self.episode < self.cur_eps_threshold:
                if d < self.initial_distance_threshold:
                    self.results.pop()
                    self.results.append(1)
                    return True
                else:
                    return False
            else:
                if d < self.final_distance_threshold:
                    self.results.pop()
                    self.results.append(1)
                    return True
                else:
                    return False
        else:
            if d < self.distance_threshold:
                self.results.pop()
                self.results.append(1)
                return True
            else:
                return False

    def _is_failure(self, achieved_goal, desired_goal):
        #d = goal_distance(achieved_goal, desired_goal)
        #return (d > self.fail_threshold) & (numpy.round(self.sim.get_state()[0]/0.0005).astype('int') > 200) # removed early stop because baselines did not work with it
        return False

    def _env_setup(self, initial_qpos):
        self.sim.data.ctrl[:] = initial_qpos
        self.set_state(initial_qpos)
        self.sim.forward()

    def render(self, mode='human', width=500, height=500):
        return super(Ur10Env, self).render(mode, width, height)