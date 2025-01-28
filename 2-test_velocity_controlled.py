"""testing the end efector movement with velocity control """


import pybullet as p
import pybullet_data
import numpy as np
import time

p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.81)

robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

start_pos = np.array([0, 0, 1])  # start
goal_pos = np.array([1, 1, 1])  # end
velocity = 0.001  #  (m/s)

end_effector_index = 6

# lineara interpolation
num_steps = int(np.linalg.norm(goal_pos - start_pos) / velocity)
t = np.linspace(0, 1, num_steps)
trajectory = (1 - t)[:, None] * start_pos + t[:, None] * goal_pos

# simu
for target_pos in trajectory:
    # Ters kinematik: uç eleman pozisyonuna göre eklem açılarını hesapla
    joint_positions = p.calculateInverseKinematics(robot_id, end_effector_index, target_pos)

    # each joint different velocity
    for i, joint_pos in enumerate(joint_positions):
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=i,
            controlMode=p.VELOCITY_CONTROL,  # controlled
            targetVelocity=(joint_pos - p.getJointState(robot_id, i)[0]) * 240.0, #atrgetVelocity=(targetPosition−currentPosition)×sim freq
            #freq-> step per sec
            force=50.0  #  max force, for remain stability
        )

    p.stepSimulation()
    time.sleep(1 / 240.0)

# Simülasyonu kapat
p.disconnect()
