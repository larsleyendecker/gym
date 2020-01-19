import numpy as np
import numpy
import mujoco_py
import json
from gym.envs.robotics import rotations, robot_env, utils
import os


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class Ur5Env(robot_env.RobotEnv):
    """Superclass for all Ur5 environments.
    """

    def __init__(
        self, model_path, n_substeps, distance_threshold, initial_qpos, reward_type,
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

        super(Ur5Env, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=6,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
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

        action *= 0.05  # limit maximum change in position

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        force = self.sim.data.sensordata
        x_pos = self.sim.data.get_body_xpos("gripper_dummy_heg")
        x_quat = self.sim.data.get_body_xquat("gripper_dummy_heg")
        q_pos = self.sim.data.qpos

        obs = np.concatenate([
            x_pos, x_quat, q_pos, force
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': np.concatenate([x_pos, x_quat]).copy(),
            'desired_goal': self.goal.copy(),
        }

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
        qpos = np.array([1.5708, 0, -1.5708, 1.5708, 1.5708, 1.5708])
        self.sim.data.ctrl[:] = qpos
        self.set_state(qpos)
        self.sim.forward()
        return True

    def _sample_goal(self):
        home_path = os.getenv("HOME")
        goal_path = os.path.join(*[home_path, "DRL_SetBot-RearVentilation", "experiment_configs", "goal.json"])

        with open(goal_path, encoding='utf-8') as file:
            goal = json.load(file)
        return numpy.concatenate([goal['xpos'], goal['xquat']]).copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        qpos = np.array([1.5708, 0, -1.5708, 1.5708, 1.5708, 1.5708])
        self.sim.data.ctrl[:] = qpos
        self.set_state(qpos)
        self.sim.forward()


    def render(self, mode='human', width=500, height=500):
        return super(Ur5Env, self).render(mode, width, height)
