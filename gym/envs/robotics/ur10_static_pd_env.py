import os
import numpy
import pandas
import mujoco_py
import json
import yaml

from gym.envs.robotics import rotations, robot_custom_env, utils
from gym.envs.robotics.ur10 import randomize

from scipy.signal import lfilter, lfilter_zi, butter

PROJECT_PATH = os.path.join(*[os.getenv("HOME"), "DRL_AI4RoMoCo"])
MODEL_PATH = os.path.join(*[PROJECT_PATH, "code", "environment", "UR10"])
CONFIG_PATH = os.path.join(*[PROJECT_PATH, "code", "config", "environment"])
SAVE_PATH = os.path.join(*[PROJECT_PATH, "code", "data", "sim_poses"])

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

class Ur10Env(robot_custom_env.RobotEnv):
    """Superclass for all Ur10 environments."""

    def __init__(self, env_config):

        with open(env_config) as cfg:
            env_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # Parameter for Custom Savings
        self.save_data = False
        self.episode = 0
        self.rewards = []
        self.distances = []
        self.sim_poses = []

        ###########################

        self.model_path = os.path.join(*[MODEL_PATH,env_config["model_xml_file"]])  # Path to the environment xml file
        self.initial_qpos = numpy.array(env_config["initial_qpos"])                 # An array of values that define the initial configuration)
        self.sim_ctrl_q = self.initial_qpos                                         
        self.reward_type = env_config["reward_type"]                                # The reward type i.e. sparse or dense
        self.ctrl_type = env_config["ctrl_type"]                            
        self.n_substeps = env_config["n_substeps"]                                  # Number of substeps the simulation runs on every call to step
        self.distance_threshold = env_config["distance_threshold"]                  # The threshold after which a goal is considered achieved
        self.fail_threshold = env_config["fail_threshold"]                          
        self.n_actions = env_config["n_actions"]
        self.corrective = env_config["corrective"]
        self.vary = env_config["vary"]
        self.dx_max = env_config["dx_max"]

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

        ############################
        
        super(Ur10Env, self).__init__(
            model_path=self.model_path, n_substeps=self.n_substeps, n_actions=self.n_actions,
            initial_qpos=self.initial_qpos)
               
    # GoalEnv methods

    def set_force_for_q(self, q_ctrl, only_grav_comp=False):
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
        '''Compute distance between goal and the achieved goal.'''
        d = goal_distance(obs, goal)

        if self.reward_type == 'sparse':
            #if (d > self.fail_threshold).astype(numpy.float32):
            #     #-8000+2n_t  ... sim.get_state()[0]/0.0005 = n_substeps * n_t
            #    reward = -8000 + numpy.round(self.sim.get_state()[0]/0.0005).astype('int')
            return -(d > self.distance_threshold).astype(numpy.float32) - 10*(d > self.fail_threshold).astype(numpy.float)
        else:
            self.rewards.append(-d)
            return -d

    # RobotEnv methods

    def _step_callback(self):
        a = 0
        # not implemented
        
    def set_state(self, qpos):
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, old_state.qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def _set_action(self, action):

        assert action.shape == (6,)
        action = action.copy()  # ensure that we don't change the action outside of this scope

        deviation = sum(abs(self.sim.data.qpos - self.sim_ctrl_q))
        
        if  deviation > 0.35:  # reset control to current position if deviation too high
            self.set_force_for_q(self.sim.data.qpos)
            #print('deviation compensated')

        if self.ctrl_type == "joint":
            action *= 0.05  # limit maximum change in position
            # Apply action #scalarsto simulation.
            utils.ctrl_set_action(self.sim, action)
        elif self.ctrl_type == "cartesian":
            dx = action.reshape(6, )

            max_limit = self.dx_max
            # limitation of operation space, we only allow small rotations adjustments in x and z directions, moving in y direction
            x_now = numpy.concatenate((self.sim.data.get_body_xpos("gripper_dummy_heg"), self.sim.data.get_body_xquat("gripper_dummy_heg")))
            x_then = x_now[:3] + dx[:3]*max_limit

            #diff_now = numpy.array(x_now - self.init_x).reshape(7,)
            diff_then = numpy.array(x_then[:3] - self.init_x[:3])

            barriers_min = numpy.array([-0.2, -0.2,   -0.2])
            barriers_max = numpy.array([0.2,  0.4, 0.2])

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
            dx_ = numpy.concatenate([rot_mat.dot(dx[:3]), rot_mat.dot(dx[3:])])  ## transform to right coordinate system
            #dx_[2]+= 1
            dq = self.get_dq(dx_)
            q = self.sim_ctrl_q + dq
            self.set_force_for_q(q)
            # print(sum(abs(sim.data.qpos-sim.data.ctrl)))


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

        x_pos = self.sim.data.get_body_xpos("gripper_dummy_heg")
        x_mat = self.sim.data.get_body_xmat("gripper_dummy_heg")
        rpy = normalize_rad(rotations.mat2euler(x_mat))

        obs = numpy.concatenate([
            rot_mat.dot(x_pos-self.goal[:3]), rot_mat.dot(normalize_rad(rpy-self.goal[3:])), ft.copy()
        ])
        #self.sim_poses.append(numpy.concatenate([x_pos-self.goal[:3], normalize_rad(rpy-self.goal[3:])]))
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
        # Visualize target.
        a=0
    
    def _reset_sim(self):
        
        if self.save_data == True and self.episode > 0:
        #    sim_poses_df = pandas.DataFrame(self.sim_poses, columns=["x", "y", "z", "rx", "ry", "rz"])
            sim_distance_001_df = pandas.DataFrame(self.distances, columns=["Distance"])
            sim_rewards_001_df = pandas.DataFrame(self.rewards, columns=["reward"])
            sim_rewards_001_df.to_feather(os.path.join(*[SAVE_PATH, "sim_rewards_000_{}.ftr".format(self.episode)]))
            sim_distance_001_df.to_feather(os.path.join(*[SAVE_PATH, "sim_distances_000_{}.ftr".format(self.episode)]))
        #    sim_poses_000_df.to_feather(os.path.join(*[SAVE_PATH, "sim_poses_000_{}.ftr".format(self.episode)]))
        #self.sim_poses = []
        self.rewards = []
        self.distances = []
        self.episode += 1
        
        #if not self.viewer is None:
        #    self._get_viewer('human').update_sim(self.sim)
        #self._get_viewer('human')._ncam = self.sim.model.ncam
        
        deviation_q = numpy.array([0, 0, 0, 0, 0, 0])

        self.ctrl_buffer = numpy.repeat((self.initial_qpos + deviation_q).reshape(1, 6), 4, axis=0)
        self.b, self.a = butter(2, 0.12)
        self.zi = [lfilter_zi(self.b, self.a) * (self.initial_qpos[i] + deviation_q[i]) for i in range(6)]
        self.qi_diff = 0
        self.last_target_q = self.initial_qpos + deviation_q

        self.set_state(self.initial_qpos + deviation_q)
        self.sim.forward()
        #self.sim.step()
        
        # Checking if there are contacts in simulation and returning false if so
        if self.sim.data.ncon == 0:
            for i in range(100):
                self.sim.data.ctrl[:] = self.sim.data.qfrc_bias.copy()
                self.sim.forward()
                #self.sim.step()
            self.init_x = numpy.concatenate((self.sim.data.get_body_xpos("gripper_dummy_heg"),
                                             self.sim.data.get_body_xquat("gripper_dummy_heg")))
            self.set_force_for_q(self.initial_qpos + deviation_q)

        return self.sim.data.ncon == 0

    def _sample_goal(self):
        home_path = os.getenv("HOME")
        goal_path = os.path.join(*[home_path, "DRL_AI4RoMoCo","code", "environment", "experiment_configs", "goal_ur10_simpheg_conf2.json"])

        with open(goal_path, encoding='utf-8') as file:
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
            rot_mat.dot(x_pos-self.goal[:3]), rot_mat.dot(normalize_rad(rpy-self.goal[3:]))
        ])
        d = goal_distance(obs, desired_goal)
        self.distances.append(d)
        return (d < self.distance_threshold).astype(numpy.float32)

    def _is_failure(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        #return (d > self.fail_threshold) & (numpy.round(self.sim.get_state()[0]/0.0005).astype('int') > 200) # removed early stop because baselines did not work with it
        return False

    def _env_setup(self, initial_qpos):
        self.sim.data.ctrl[:] = initial_qpos
        self.set_state(initial_qpos)
        self.sim.forward()


    def render(self, mode='human', width=500, height=500):
        return super(Ur10Env, self).render(mode, width, height)