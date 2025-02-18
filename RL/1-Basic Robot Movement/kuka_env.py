import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
from gymnasium import spaces


class KukaEnv(gym.Env):
    def __init__(self):
        super(KukaEnv, self).__init__()

        # PyBullet connection
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # for kuka load
        self.reset()

        #Action space: (Continuous)**: For 7 joints between [-1, 1] continuous actions
        #tanh(x) inputs for PPO agent activation and it requires [-1,1]-> helps more stabilizated learning
        #the joint limits are taken as [-90,90] degrees, so optimization in this gap wpuld be more covenient
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)

        # Observation Space**: Robot joint+ end efector positions
        # 7 degrees of freedom and 3 for end efector position(x-y-z orientation)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(10,), dtype=np.float32
        )


#for each training period, in the start to take the agent to initial and reset the environment
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(self.physics_client)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

        # taking robot to the initial

        self.initial_joint_positions = [0] * 7
        for i in range(7):
            p.resetJointState(self.robot_id, i, self.initial_joint_positions[i])

        obs = self._get_observation()

        info = {}

        return obs, info

    def step(self, action):

        for i in range(7):
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, action[i])

        # continues withb the next step of the simulation
        p.stepSimulation()

        # get new statues
        obs = self._get_observation()

        ############ REWARD FUNCTION ############

        #job: End efector gets close to a target
        end_effector_pos = self._get_end_effector_position()
        target_pos = [0.5, 0, 0.3]
        distance_to_target = np.linalg.norm(np.array(end_effector_pos) - np.array(target_pos))

        ############ REWARD CALCULATIONS ############
        reward = -distance_to_target

        #if it's so close big reward
        terminated = distance_to_target < 0.008
        truncated = False  # for time limit

        info = {}

        return obs, reward, terminated, truncated, info

    # downwards two func gets the current situations

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
