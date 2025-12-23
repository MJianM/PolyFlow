
# %%%

import os
import contextlib
import subprocess

import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import numpy as np

from minari import DataCollector

from gymnasium.wrappers import RecordVideo

@contextlib.contextmanager
def virtual_display(display_id=":99", screen_size="1024x768x24"):
    """虚拟显示上下文管理器"""
    xvfb_process = None
    original_display = os.environ.get("DISPLAY")
    
    try:
        if "DISPLAY" not in os.environ:
            print(f"启动Xvfb虚拟显示 {display_id}...")
            xvfb_process = subprocess.Popen(
                ["Xvfb", display_id, "-screen", "0", screen_size]
            )
            os.environ["DISPLAY"] = display_id
            # 给Xvfb一点启动时间
            import time
            time.sleep(1)
        
        yield  # 这里执行主程序代码
        
    finally:
        # 清理：关闭Xvfb并恢复原来的DISPLAY设置
        if xvfb_process is not None and xvfb_process.poll() is None:
            print("关闭Xvfb进程...")
            xvfb_process.terminate()
            xvfb_process.wait()
        
        # 恢复原来的DISPLAY环境变量
        if original_display is not None:
            os.environ["DISPLAY"] = original_display
        elif "DISPLAY" in os.environ:
            del os.environ["DISPLAY"]

def set_all_seed(seed):
    np.random.seed(seed)

# %%
# WayPoint Planner
# ~~~~~~~~~~~~~~~~
# Our first task is to create a method that generates a trajectory to the goal in the maze.
# We have the advantage that the MuJoCo maze can be discretized into a grid of cells, which reduces
# the size of the state space. The action space for this solver will also be reduced to ``UP``, ``DOWN``, ``LEFT``,
# and ``RIGHT``. The solution trajectories will then be a set of waypoints that the agent has to follow to reach the
# goal.
# We can simply use a variation of Dynamic Programming to generate the trajectories. The method chosen in the D4RL[1] publication
# is that of Value Iteration, specifically Q-Value Iteration[2]. We will obtain the optimal Q-values by doing a series of Bellman
# updates (``50`` in total) of the form:
#
# .. math::
#   Q'(s, a) \leftarrow \sum_{s'}T(s,a,s')[R(s,a,s') + \gamma\max_{a'}Q(s',a')]
#
# **T(s,a,s')** is the transition matrix which gives the probability of reaching state **s'** when taking action **a** from state **s**.
# We consider the grid maze a deterministic space which means that if **s'** is an empty cell **T(s,a,s')** will have a value of ``1`` since
# we know that the agent will always reach that state. On the other hand, if the state **s'** is a wall the value of **T(s,a,s')** will be ``0``.
#
# Once we have the optimal Q-values (**Q***) we can generate a waypoint trajectory with the following policy:
#
# .. math::
#   \pi(s) = arg\max_{a}Q^{*}(s,a)
#
# The class below, ``QIteration``, gives access to the method ``generate_path(current_cell, goal_cell)``.  This method returns a dictionary of waypoints
# such as:
#
# .. code:: py
#
#   {(5, 1): (4, 1), (4, 1): (4, 2), (4, 2): (3, 2), (3, 2): (2, 2), (2, 2): (2, 1), (2, 1): (1, 1)}
#
# The keys of this dictionary are the current state of the agent and the values the next state of the wapoint path.

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

EXPLORATION_ACTIONS = {UP: (0, 1), DOWN: (0, -1), LEFT: (-1, 0), RIGHT: (1, 0)}


class QIteration:
    """Solves for optimal policy with Q-Value Iteration.

    Inspired by https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/pointmaze/q_iteration.py
    """

    def __init__(self, maze):
        self.maze = maze
        self.num_states = maze.map_length * maze.map_width
        self.num_actions = len(EXPLORATION_ACTIONS.keys())
        self.rew_matrix = np.zeros((self.num_states, self.num_actions))
        self.compute_transition_matrix()

    def generate_path(self, current_cell, goal_cell):
        self.compute_reward_matrix(goal_cell)
        q_values = self.get_q_values()
        current_state = self.cell_to_state(current_cell)
        waypoints = {}
        while True:
            action_id = np.argmax(q_values[current_state])
            next_state, _ = self.get_next_state(
                current_state, EXPLORATION_ACTIONS[action_id]
            )
            current_cell = self.state_to_cell(current_state)
            waypoints[current_cell] = self.state_to_cell(next_state)
            if waypoints[current_cell] == goal_cell:
                break

            current_state = next_state

        return waypoints

    def reward_function(self, desired_cell, current_cell):
        if desired_cell == current_cell:
            return 1.0
        else:
            return 0.0

    def state_to_cell(self, state):
        i = int(state / self.maze.map_width)
        j = state % self.maze.map_width
        return (i, j)

    def cell_to_state(self, cell):
        return cell[0] * self.maze.map_width + cell[1]

    def get_q_values(self, num_itrs=50, discount=0.99):
        q_fn = np.zeros((self.num_states, self.num_actions))
        for _ in range(num_itrs):
            v_fn = np.max(q_fn, axis=1)
            q_fn = self.rew_matrix + discount * self.transition_matrix.dot(v_fn)
        return q_fn

    def compute_reward_matrix(self, goal_cell):
        for state in range(self.num_states):
            for action in range(self.num_actions):
                next_state, _ = self.get_next_state(state, EXPLORATION_ACTIONS[action])
                next_cell = self.state_to_cell(next_state)
                self.rew_matrix[state, action] = self.reward_function(
                    goal_cell, next_cell
                )

    def compute_transition_matrix(self):
        """Constructs this environment's transition matrix.
        Returns:
          A dS x dA x dS array where the entry transition_matrix[s, a, ns]
          corresponds to the probability of transitioning into state ns after taking
          action a from state s.
        """
        self.transition_matrix = np.zeros(
            (self.num_states, self.num_actions, self.num_states)
        )
        for state in range(self.num_states):
            for action_idx, action in EXPLORATION_ACTIONS.items():
                next_state, valid = self.get_next_state(state, action)
                if valid:
                    self.transition_matrix[state, action_idx, next_state] = 1

    def get_next_state(self, state, action):
        cell = self.state_to_cell(state)

        next_cell = tuple(map(lambda i, j: int(i + j), cell, action))
        next_state = self.cell_to_state(next_cell)

        return next_state, self._check_valid_cell(next_cell)

    def _check_valid_cell(self, cell):
        # Out of map bounds
        if cell[0] >= self.maze.map_length:
            return False
        elif cell[1] >= self.maze.map_width:
            return False
        # Wall collision
        elif self.maze.maze_map[cell[0]][cell[1]] == 1:
            return False
        else:
            return True


# %%
# Waypoint Controller
# ~~~~~~~~~~~~~~~~~~~
# The step will be to create a controller to allow the agent to follow the waypoint trajectory.
# D4RL uses a PD controller to output continuous force actions from position and velocity.
# A PD controller is a variation of the PID controller often used in classical Control Theory.
# PID combines three components: a Proportial Term(P), Integral Term(I) and Derivative Term (D)
#
# 1. Proportional Term (P)
# -------------------
# The proportional term in a PID controller adjusts the control action based on the current error, which
# is the difference between the desired value (setpoint) and the current value of the process variable.
# The control action is directly proportional to the error. A higher error results in a stronger control action.
# However, the proportional term alone can lead to overshooting or instability. Note :math:`\tau` is our control value.
#
# .. math ::
#   \tau = k_{p}(\text{Error})
#
# 2. Derivative Term (D)
# -------------------
# The derivative term in a PD controller considers the rate of change of the error over time.
# It helps to predict the future behavior of the error. By dampening the control action based
# on the rate of change of the error, the derivative term contributes to system stability and reduces overshooting.
# It also helps the system respond quickly to changes in the error.
#
# .. math ::
#   \tau = k_{d}(d(\text{Error}) / dt)
#
# So for a PD controller we have the equation below. We explain what the values :math:`k_{d}` and :math:`k_{p}` mean in a bit
#
# .. math ::
#   \tau = k_{p}(\text{Error})  + k_{d}(d(\text{Error}) / dt)
#
# 3. Integral Term (I)
# -------------------
# The integral term in a PID controller integrates the cumulative error over time.
# It helps to address steady-state errors or biases that may exist in the system.
# The integral term continuously adjusts the control action based on the accumulated error,
# aiming to eliminate any long-term deviations between the desired setpoint and the actual process variable.
#
# .. math ::
#   \tau = k_{I}{\int}_0^t(\text{Error}) dt
#
# Finally for a PID controller we have the equation below
#
# .. math ::
#   \tau = k_{p}(\text{Error})  + k_{d}(d(\text{Error}) / dt) +  k_{I}\int_{0}^{t}(\text{Error}) dt
#
# In the PID controller formula, :math:`K_p`, :math:`K_i`, and :math:`K_d` are the respective gains for the proportional, integral, and derivative terms.
# These gains determine the influence of each term on the control action.
# The optimal values for these gains are typically determined through tuning, which involves adjusting
# the gains to achieve the desired control performance.
#
# Now back to our controller as stated previously, for the D4RL task we use a PD controller and we
# follow the same theme as what we have stated before as can be seen below. The :math:`Error` is equlivalent
# to the difference between the :math:`\text{goal}_\text{pose}` and :math:`\text{agent}_\text{pose}` and we replace the derivative term :math:`(d(\text{Error}) / dt)` with
# the velocity of the the agent :math:`v_{\text{agent}}`, we can think of this as a measure of the speed at which the agent
# is approaching the target position. When the agent is moving quickly towards the target, the
# derivative term will be larger, contributing to a stronger corrective action from the controller.
# On the other hand, if the agent is already close to the target and moving slowly, the derivative term will be smaller,
# resulting in a less aggressive control action.
#
# .. math ::
#   \tau = k_{p}(p_{\text{goal}} - p_{\text{agent}}) + k_{d}v_{\text{agent}}
#
# Each target position in the waypoint trajectory is converted from discrete to a continuous value and we also add some noise to
# the :math:`x` and :math:`y` coordinates to add more variance in the trajectories generated for the offline dataset.


class WaypointController:
    """Agent controller to follow waypoints in the maze.

    Inspired by https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/pointmaze/waypoint_controller.py
    """

    def __init__(self, maze, gains={"p": 10.0, "d": -1.0}, waypoint_threshold=0.1):
        self.global_target_xy = np.empty(2)
        self.maze = maze

        self.maze_solver = QIteration(maze=self.maze)

        self.gains = gains
        self.waypoint_threshold = waypoint_threshold
        self.waypoint_targets = None

    def compute_action(self, obs):
        # Check if we need to generate new waypoint path due to change in global target
        if (
            np.linalg.norm(self.global_target_xy - obs["desired_goal"]) > 1e-3
            or self.waypoint_targets is None
        ):
            # Convert xy to cell id
            achieved_goal_cell = tuple(
                self.maze.cell_xy_to_rowcol(obs["achieved_goal"])
            )
            self.global_target_id = tuple(
                self.maze.cell_xy_to_rowcol(obs["desired_goal"])
            )
            self.global_target_xy = obs["desired_goal"]

            self.waypoint_targets = self.maze_solver.generate_path(
                achieved_goal_cell, self.global_target_id
            )

            # Check if the waypoint dictionary is empty
            # If empty then the ball is already in the target cell location
            if self.waypoint_targets:
                self.current_control_target_id = self.waypoint_targets[
                    achieved_goal_cell
                ]
                self.current_control_target_xy = self.maze.cell_rowcol_to_xy(
                    np.array(self.current_control_target_id)
                )
            else:
                self.waypoint_targets[
                    self.current_control_target_id
                ] = self.current_control_target_id
                self.current_control_target_id = self.global_target_id
                self.current_control_target_xy = self.global_target_xy

        # Check if we need to go to the next waypoint
        dist = np.linalg.norm(self.current_control_target_xy - obs["achieved_goal"])
        if (
            dist <= self.waypoint_threshold
            and self.current_control_target_id != self.global_target_id
        ):
            self.current_control_target_id = self.waypoint_targets[
                self.current_control_target_id
            ]
            # If target is global goal go directly to goal position
            if self.current_control_target_id == self.global_target_id:
                self.current_control_target_xy = self.global_target_xy
            else:
                self.current_control_target_xy = (
                    self.maze.cell_rowcol_to_xy(
                        np.array(self.current_control_target_id)
                    )
                    - np.random.uniform(size=(2,)) * 0.2
                )

        action = (
            self.gains["p"] * (self.current_control_target_xy - obs["achieved_goal"])
            + self.gains["d"] * obs["observation"][2:]
        )
        action = np.clip(action, -1, 1)

        return action


# %%
# Collect Data and Create Minari Dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LARGE_MAZE =   [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 'g', 0, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 1, 0, 'r', 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]




def collect_data():

    dataset_name = "pointmaze/custom-v0"
    max_episode_steps = 1000
    total_episodes = 1000
    seed = 42
    set_all_seed(seed)

    gym.register_envs(gymnasium_robotics)
    env = gym.make('PointMaze_Large-v3', maze_map=LARGE_MAZE, continuing_task=False, reset_target=False, max_episode_steps=max_episode_steps,
                render_mode='rgb_array')

    env = RecordVideo(
        env=env,
        video_folder="./maze_videos",
        episode_trigger=lambda x: x>=0 and x<5, # 只录制前5条episode
        disable_logger=True
    )

    env = DataCollector(
        env, record_infos=True
    )

    waypoint_controller = WaypointController(maze=env.unwrapped.maze)

    truncated_cnt = 0

    for ep in range(total_episodes):
        obs, _ = env.reset()
        print("inital pos: ", obs['observation'][0:2])
        print("target: ", obs['desired_goal'][0:2])
        while True:
            action = waypoint_controller.compute_action(obs)
            # Add some noise to each step action
            action += np.random.randn(*action.shape) * 0.4
            action = np.clip(
                action, env.action_space.low, env.action_space.high, dtype=np.float32
            )

            obs, rew, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if truncated:
                    truncated_cnt += 1
                break

    print("total episode:", total_episodes)
    print("truncated count:", truncated_cnt)

    dataset = env.create_dataset(
        dataset_id=dataset_name,
        algorithm_name="QIteration",
        code_permalink="https://github.com/Farama-Foundation/Minari/blob/main/docs/tutorials/dataset_creation/point_maze_dataset.py",
        author="mjm",
        author_email="nymjm11@sjtu.edu.cn",
    )

    # 关闭环境确保视频保存
    env.close()
    print("视频已保存至 ./maze_videos 文件夹")


if __name__=="__main__":

    with virtual_display():
        collect_data()