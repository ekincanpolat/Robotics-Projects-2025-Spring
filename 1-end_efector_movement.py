"""For moving the end efector- in basic level"""

import pybullet as p
import pybullet_data
import numpy as np
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

start_pos = np.array([0, 0, 1])  # start point
goal_pos = np.array([1, 1,1])  # destination
velocity = 0.001  # (m/s)

# index for the ee
end_effector_index = 6

# linear interpolation: setting some points and timer
num_steps = int(np.linalg.norm(goal_pos - start_pos) / velocity)
t = np.linspace(0, 1, num_steps)
trajectory = (1 - t)[:, None] * start_pos + t[:, None] * goal_pos

# simulation
for target_pos in trajectory:
    # inverse kin* takes the end pos and calculates the joint degrees
    joint_positions = p.calculateInverseKinematics(robot_id, end_effector_index, target_pos)

    # Joint positions- sets joint to the desired degrees
    for i in range(len(joint_positions)):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=joint_positions[i])


    p.stepSimulation()
    time.sleep(1 / 240.0)

# Simülasyonu kapat
p.disconnect()
