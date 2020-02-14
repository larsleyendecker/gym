import numpy as np
import numpy
import mujoco_py
import json
from gym.envs.robotics import rotations, robot_custom_env, utils
import os
import matplotlib.pyplot as plt

def goal_distance(obs, goal):
    obs = obs[:6]
    assert obs.shape == goal.shape
    return np.linalg.norm(obs*np.array([1, 1, 1, 0.3, 0.3, 0.3]), axis=-1)



class Ur10Env(robot_custom_env.RobotEnv):
    """Superclass for all Ur10 environments.
    """

    def __init__(
        self, model_path, n_substeps, distance_threshold, initial_qpos, reward_type, ctrl_type="joint",
            fail_threshold=0.25
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (array): an array of values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.ctrl_type = ctrl_type
        self.fail_threshold = fail_threshold

        super(Ur10Env, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=6,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

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
            return -d

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

        if self.ctrl_type == "joint":
            action *= 0.05  # limit maximum change in position
            # Apply action to simulation.
            utils.ctrl_set_action(self.sim, action)
        elif self.ctrl_type == "cartesian":
            dx = action.reshape(6, )

            max_limit = 0.00005
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
                dx[i] = dx[i] * max_limit * 0.01 #slower rotations

            # dx[1] += 0.5*max_limit #bias in direction of assembly
            dx.reshape(6, 1)

            jacp = self.sim.data.get_body_jacp(name="gripper_dummy_heg").reshape(3, 6)
            jacr = self.sim.data.get_body_jacr(name="gripper_dummy_heg").reshape(3, 6)
            jac = numpy.vstack((jacp, jacr))
            dq = numpy.linalg.lstsq(jac, dx)[0].reshape(6, )
            # print(sum(abs(sim.data.qpos-sim.data.ctrl)))
            for i in range(6):
                self.sim.data.ctrl[i] += dq[i]


    def _get_obs(self):
        # positions
        force = self.sim.data.sensordata
        x_pos = self.sim.data.get_body_xpos("gripper_dummy_heg")
        x_quat = self.sim.data.get_body_xquat("gripper_dummy_heg")
        rpy = rotations.quat2euler(x_quat)

        obs = np.concatenate([
            x_pos-self.goal[:3], rpy-self.goal[3:], force
        ])

        return obs
        '''
        return {
            'observation': obs.copy(),
            'achieved_goal': np.concatenate([x_pos, rpy]).copy(),
            'desired_goal': self.goal.copy(),
        }
        '''

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
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        self.init_x = numpy.concatenate((self.sim.data.get_body_xpos("gripper_dummy_heg"), self.sim.data.get_body_xquat("gripper_dummy_heg")))
        self.sim.data.ctrl[:] = self.initial_state[1]
        #self.set_state(qpos)
        #self.sim.forward()
        return True

    def _sample_goal(self):
        home_path = os.getenv("HOME")
        goal_path = os.path.join(*[home_path, "DRL_SetBot-RearVentilation", "experiment_configs", "goal_ur10.json"])

        with open(goal_path, encoding='utf-8') as file:
            goal = json.load(file)
            xpos =  goal['xpos']
            xquat = goal['xquat']
            rpy = rotations.quat2euler(xquat)
        return numpy.concatenate([xpos, rpy]).copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
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
