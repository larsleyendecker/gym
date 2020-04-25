import numpy as np
import numpy
import mujoco_py
import json
from gym.envs.robotics import rotations, robot_custom_env, utils
import os
from gym.envs.robotics.ur10 import randomize
import matplotlib.pyplot as plt
from scipy.signal import lfilter, lfilter_zi, butter

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

    def __init__(
        self, model_path, n_substeps, distance_threshold, initial_qpos, reward_type, ctrl_type="joint",
            fail_threshold=0.25, vary=False,
            init_vary_range=numpy.array([0.03, 0.03, 0.03, 3/180*np.pi, 3/180*np.pi, 3/180*np.pi]), corrective=False,
            worker_id=1, randomize_kwargs={}, pos_std=numpy.array([0, 0, 0, 0, 0, 0]),
            pos_drift_range=numpy.array([0, 0, 0, 0, 0, 0]), ft_noise=False, ft_drift=False, punish_force=False,
            punish_force_thresh=15,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (array): an array of values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.xml_path = model_path
        self.offset = randomize.randomize_ur10_xml()
        self.n_substeps = n_substeps
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.ctrl_type = ctrl_type
        self.fail_threshold = fail_threshold
        self.vary = vary
        self.initial_qpos = initial_qpos
        self.corrective = corrective
        self.sim_ctrl_q = initial_qpos
        self.worker_id = worker_id
        self.punish_force = punish_force
        self.punish_force_thresh = punish_force_thresh
        self.init_vary_range = init_vary_range

        self.pos_std = pos_std
        self.pos_drift_range = pos_drift_range
        self.pos_drift_val = numpy.zeros(6,)   # set in sim reset function

        self.ft_noise = ft_noise
        self.ft_drift = ft_drift
        self.ft_std = numpy.array([0.1, 0.1, 0.1, 0.003, 0.003, 0.003]) #These values are based on measurements
        self.ft_drift_range = numpy.array([1, 1, 1, 0.015, 0.015, 0.015]) #and these ,too
        self.ft_drift_val = 0    # set in sim reset function


        self.K = numpy.array([14.87, 13.26, 11.13, 10.49, 11.03, 11.47])
        self.kp = numpy.array([4247, 3342, 3306, 707, 1236, 748])
        self.ki = numpy.array([70.218, 38.65, 86.12, 19.60, 17.07, 19.40])
        self.kd = numpy.array([0, 0, 0, 0, 0, 0])  # numpy.array([81.61, 77.38, 10.61, 10.44, 4.75, 9.14])
        self.kf = numpy.array([28, 16.7, 8.42, 2.42, 4.12, 2.34])
        self.max_fb = numpy.array([7.18, 1.54, 4.82, 3.22, 1.41, 1.891])

        self.ctrl_buffer = np.repeat(initial_qpos.copy().reshape(1, 6), 4, axis=0)
        b, a = butter(2, 0.12)
        self.a = a
        self.b = b
        self.zi = [lfilter_zi(b, a) * initial_qpos[i] for i in range(6)]
        self.qi_diff = 0
        self.last_target_q = initial_qpos.copy()

        self.randomize_kwargs = randomize_kwargs
        super(Ur10Env, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=6,
            initial_qpos=initial_qpos)
        self.p = [900,
             8000,
             700,
             100,
             100,
             100]
        self.d = [12,
             80,
             5,
             0.5,
             0.5,
             0.5, ]
        self.max_T = np.array(
            [40,
             50,
             35,
             2,
             2,
             2])
    # GoalEnv methods
    # ----------------------------


    def set_force_for_q(self, q_ctrl, only_grav_comp=False):
        q_ctrl = q_ctrl.reshape(6, )

        self.ctrl_buffer[1:] = self.ctrl_buffer[:3].copy()
        self.ctrl_buffer[0] = q_ctrl.copy()
        target_q = np.zeros(6)

        for i in range(6):
            target_q[i], zo = lfilter(self.b, self.a, [self.ctrl_buffer[3][i]], zi=self.zi[i])
            self.zi[i] = zo.copy()

        target_qd = (target_q - self.last_target_q).copy() * 125
        self.last_target_q = target_q.copy()

        q_diff = target_q - self.sim.data.qpos
        qd_diff = target_qd - self.sim.data.qvel
        self.qi_diff += q_diff

        #for j in range(6):
        #    if np.sign(self.qi_diff[j]) != np.sign(q_diff[j]):
        #        self.qi_diff[j] = 0

        self.sim.data.ctrl[:] = self.sim.data.qfrc_bias.copy()

        if not only_grav_comp:
            self.sim.data.ctrl[:] += self.K * (np.clip(self.kp * q_diff + self.ki * self.qi_diff,
                                                       -self.max_fb, self.max_fb) + self.kf * target_qd)
        self.sim_ctrl_q = q_ctrl
        return self.sim.data.ctrl[:].copy()

    #def set_force_for_q(self, q, only_grav_comp=False):
    #    self.sim.data.ctrl[:] = self.sim.data.qfrc_bias  # + sim.data.qfrc_bias*0.01*numpy.random.randn(6,)
    #    if not only_grav_comp:
    #        self.sim.data.ctrl[:] += np.clip(self.p * (q - self.sim.data.qpos) + self.d * (0 - self.sim.data.qvel), -self.max_T, self.max_T)
    #    self.sim_ctrl_q = q
    #    return self.sim.data.ctrl[:].copy()


    def activate_noise(self):
        self.vary=True
        print('noise has been activated.')

    def compute_reward(self, obs, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(obs, goal)
        #print(d)
        if self.reward_type == 'sparse':
            #if (d > self.fail_threshold).astype(np.float32):
            #     #-8000+2n_t  ... sim.get_state()[0]/0.0005 = n_substeps * n_t
            #    reward = -8000 + numpy.round(self.sim.get_state()[0]/0.0005).astype('int')
            return -(d > self.distance_threshold).astype(np.float32) - 10*(d > self.fail_threshold).astype(np.float)
        else:
            rew = -d
            force_amp = numpy.linalg.norm(obs[6:9])
            if self.punish_force and force_amp > self.punish_force_thresh:
                rew -= 0.1 * force_amp

            return rew

    # RobotEnv methods
    # ----------------------------

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
        # print(deviation )
        if  deviation > 0.35:  # reset control to current position if deviation too high
            self.set_force_for_q(self.sim.data.qpos)
            #print('deviation compensated')

        if self.ctrl_type == "joint":
            action *= 0.05  # limit maximum change in position
            # Apply action #scalarsto simulation.
            utils.ctrl_set_action(self.sim, action)
        elif self.ctrl_type == "cartesian":
            dx = action.reshape(6, )

            max_limit = 0.001
            # limitation of operation space, we only allow small rotations adjustments in x and z directions, moving in y direction
            x_now = numpy.concatenate((self.sim.data.get_body_xpos("gripper_dummy_heg"), self.sim.data.get_body_xquat("gripper_dummy_heg")))
            x_then = x_now[:3] + dx[:3]*max_limit

            #diff_now = numpy.array(x_now - self.init_x).reshape(7,)
            diff_then = numpy.array(x_then[:3] - self.init_x[:3])

            barriers_min = numpy.array([-0.1, -0.05,   -0.1])
            barriers_max = numpy.array([0.1,  0.2, 0.1])

            for i in range(3):
                if (barriers_min[i] < diff_then[i] < barriers_max[i]):
                    dx[i] = dx[i] * max_limit
                elif barriers_min[i] > diff_then[i]:
                    dx[i] = + max_limit
                elif barriers_max[i] < diff_then[i]:
                    dx[i] = - max_limit
            for i in range(3,6):
                dx[i] = dx[i] * max_limit

            if self.corrective:
                # bias in direction of assembly
                bias_dir = -self.last_obs[:6]
                # print(bias_dir)
                for i in range(3,6):
                    if bias_dir[i] > 0.5:
                        print(i, bias_dir[i])
                    bias_dir[i] = bias_dir[i] # slower rotations
                bias_dir /= np.linalg.norm(bias_dir)
                # print(bias_dir)
                dx += bias_dir * max_limit * 0.5
                dx.reshape(6, 1)

            rot_mat = self.sim.data.get_body_xmat('gripper_dummy_heg')
            dx_ = np.concatenate([rot_mat.dot(dx[:3]), rot_mat.dot(dx[3:])])  ## transform to right coordinate system
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
        if self.ft_noise:
            ft += numpy.random.randn(6,) * self.ft_std
        if self.ft_drift:
            ft += self.ft_drift_val
        ft[1] -= 0.350*9.81  # nulling in initial position
        x_pos = self.sim.data.get_body_xpos("gripper_dummy_heg")
        x_pos += numpy.random.uniform(-self.pos_std[:3], self.pos_std[:3])
        x_pos += self.pos_drift_val[:3]

        #x_quat = self.sim.data.get_body_xquat("gripper_dummy_heg")
        x_mat = self.sim.data.get_body_xmat("gripper_dummy_heg")
        rpy = normalize_rad(rotations.mat2euler(x_mat)
                            + numpy.random.uniform(-self.pos_std[3:], self.pos_std[3:])
                            + self.pos_drift_val[3:])

        obs = np.concatenate([
            rot_mat.dot(x_pos-self.goal[:3]), rot_mat.dot(normalize_rad(rpy-self.goal[3:])), ft.copy()
        ])

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
        if self.vary == True:
            deviation_x = numpy.random.uniform(-self.init_vary_range, self.init_vary_range)
            deviation_q = self.get_dq(deviation_x)
        else:
            deviation_q = numpy.array([0, 0, 0, 0, 0, 0])

        if self.ft_drift:
            self.ft_drift_val = numpy.random.uniform(-self.ft_drift_range, self.ft_drift_range)
        self.pos_drift_val = numpy.random.uniform(-self.pos_drift_range, self.pos_drift_range)

        del self.sim
        self.offset = randomize.randomize_ur10_xml(worker_id=self.worker_id, **self.randomize_kwargs)
        model = mujoco_py.load_model_from_path(self.xml_path)
        self.sim = mujoco_py.MjSim(model, nsubsteps=self.n_substeps)
        self.goal = self._sample_goal()
        if not self.viewer is None:
            self._get_viewer('human').update_sim(self.sim)
        #self._get_viewer('human')._ncam = self.sim.model.ncam

        self.ctrl_buffer = np.repeat((self.initial_qpos + deviation_q).reshape(1, 6), 4, axis=0)
        self.b, self.a = butter(2, 0.12)
        self.zi = [lfilter_zi(self.b, self.a) * (self.initial_qpos[i] + deviation_q[i]) for i in range(6)]
        self.qi_diff = 0
        self.last_target_q = self.initial_qpos + deviation_q

        self.set_state(self.initial_qpos + deviation_q)
        self.sim.forward()
        for i in range(100):
            self.sim.data.ctrl[:] = self.sim.data.qfrc_bias.copy()
            self.sim.step()
        self.init_x = numpy.concatenate((self.sim.data.get_body_xpos("gripper_dummy_heg"),
                                         self.sim.data.get_body_xquat("gripper_dummy_heg")))
        self.set_force_for_q(self.initial_qpos + deviation_q)
        return True

    def _sample_goal(self):
        home_path = os.getenv("HOME")
        goal_path = os.path.join(*[home_path, "DRL_SetBot-RearVentilation", "experiment_configs", "goal_ur10_simpheg_conf2.json"])

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
        return (d < self.distance_threshold).astype(np.float32)

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
