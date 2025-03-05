#in this code, we added joint velocity created a new reward func

import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
from gymnasium import spaces


class KukaEnv(gym.Env):
    def __init__(self):
        super(KukaEnv, self).__init__()

        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.reset()

        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)

        self.max_force = 50
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(self.physics_client)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

        self.initial_joint_positions = [0] * 7
        for i in range(7):
            p.resetJointState(self.robot_id, i, self.initial_joint_positions[i])

        return self._get_observation(), {}

    def step(self, action):
        action = np.clip(action * 2.0, -1, 1)

        # Apply velocity control with force limit
        for i in range(7):
            p.setJointMotorControl2(
                self.robot_id, i, p.VELOCITY_CONTROL, targetVelocity=action[i] * 5, force=self.max_force
            )

        p.stepSimulation()

        obs = self._get_observation()

        # Reward function: Distance-based for now
        end_effector_pos = self._get_end_effector_position()
        target_pos = [0.5, 0, 0.3]
        distance_to_target = np.linalg.norm(np.array(end_effector_pos) - np.array(target_pos))

        # calculation of the sum of the velocities
        joint_velocities = np.array([p.getJointState(self.robot_id, i)[1] for i in range(7)])
        velocity_penalty = np.sum(np.square(joint_velocities))  # q^T q

        # NEW REWARD FUNCTION

        # first component: distance
        # k1*1/(1 + distance_to_target**2)-> aim is when the distance is smaller, it contributes more to reward
        # and squaring the number to avoid negative/positive assignments from calculations

        #second component: velocities of joints
        #summing the squared velocities ensures that arger speeds are penalized more and removes the effect of direction
        #we extract that to lower the price to provice promotes smooth, controlled movements in uncontrolled movement

        k1, k2 = 10, 0  # constants
        reward = k1 * (1 / (1 + distance_to_target**2)) + k2 * velocity_penalty**2

        terminated = distance_to_target < 0.008
        truncated = False


        return obs, reward, terminated, truncated, {}

    def _get_observation(self):
        joint_states = [p.getJointState(self.robot_id, i)[0] for i in range(7)]
        end_effector_pos = self._get_end_effector_position()
        return np.array(joint_states + list(end_effector_pos), dtype=np.float32)

    def _get_end_effector_position(self):
        end_effector_index = 6
        link_state = p.getLinkState(self.robot_id, end_effector_index)
        return link_state[0]

    def close(self):
        p.disconnect(self.physics_client)
