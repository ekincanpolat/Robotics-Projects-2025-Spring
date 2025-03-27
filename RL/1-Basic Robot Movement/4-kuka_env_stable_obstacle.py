#This enviroment is aiming a stable obstacle and avoidng it while going to the aimed destination
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

        # obstacle
        self.obstacle_pos = np.array([0.3, 0.2, 0.2])

        self.reset()

        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(10,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(self.physics_client)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

        # initial joint positions
        self.initial_joint_positions = [0] * 7
        for i in range(7):
            p.resetJointState(self.robot_id, i, self.initial_joint_positions[i])

        # obstacle
        self.obstacle_id = p.loadURDF("cube_small.urdf", self.obstacle_pos, useFixedBase=True)

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        action = np.clip(action * 2.0, -1, 1)

        for i in range(7):
            p.setJointMotorControl2(
                self.robot_id, i, p.VELOCITY_CONTROL, targetVelocity=action[i]
            )

        p.stepSimulation()
        obs = self._get_observation()

        end_effector_pos = self._get_end_effector_position()
        target_pos = [0.5, 0, 0.3]
        distance_to_target = np.linalg.norm(np.array(end_effector_pos) - np.array(target_pos))

        # distance to obstacle
        obstacle_pos = np.array(self.obstacle_pos)
        distance_to_obstacle = np.linalg.norm(np.array(end_effector_pos) - obstacle_pos)

        joint_velocities = np.array([p.getJointState(self.robot_id, i)[1] for i in range(7)])

        # reward func
        k1, k2, k3, k4 = 11, 0.05, 2, 15
      #same as before this part
        reward = k1 * (1 / (1 + distance_to_target ** 2)) - k2 * np.sum(np.square(joint_velocities))
      #ADDITIONS!!
        # if it gets too close to the obstacle, punishment with a medium constant
        if distance_to_obstacle < 0.01:
            reward -= k3 * ( distance_to_obstacle)

        #if its touching the obstacle, a big punishment
        if distance_to_obstacle < 0.001:
            reward -= k4


        terminated = distance_to_target < 0.008
        truncated = False

        print(f"Step:")
        print(f"  Action: {action}")
        print(f"  Distance to the Target: {distance_to_target:.5f}")
        print(f"  Distance to the Obstacle: {distance_to_obstacle:.5f}")
        print(f"  Joint Velocities: {joint_velocities}")
        print(f"  Reward: {reward:.5f}")
        print("-" * 50)

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
