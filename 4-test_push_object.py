""" algorithm for pushing an object"""


import pybullet as p
import pybullet_data
import numpy as np
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

plane_id = p.loadURDF("plane.urdf")

robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
object_id = p.loadURDF("cube.urdf", basePosition=[1, 0, 1], globalScaling=-1.5)
wait_time = 3

object_pos, _ = p.getBasePositionAndOrientation(object_id)
touch_position = np.array(object_pos) + np.array([-0.1, 0, 0]) #selects the push location
push_position = touch_position + np.array([0.3, 0, 0])  # the push

start_position = [0, 0, 1]

velocity = 0.005
distance_to_touch = np.linalg.norm(np.array(touch_position) - np.array(start_position))
num_steps_touch = int(distance_to_touch / velocity) * 2

distance_to_push = np.linalg.norm(np.array(push_position) - np.array(touch_position))
num_steps_push = int(distance_to_push / velocity) * 2

trajectory_to_touch = np.linspace(start_position, touch_position, num_steps_touch)
trajectory_to_push = np.linspace(touch_position, push_position, num_steps_push)

print(f"Başlangıç pozisyonunda bekleniyor ({wait_time} saniye)...")
time.sleep(wait_time)

print("Cisme yaklaşılıyor...")
for step, pos in enumerate(trajectory_to_touch):
    joint_positions = p.calculateInverseKinematics(robot_id, 6, pos)

    for i in range(len(joint_positions)):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=joint_positions[i])

    p.stepSimulation()
    time.sleep(1 / 480.0)

    # Pozisyon güncellemesi
    print(f"STEP {step + 1}: Robot approaches. Target: {pos}")

print("Robot touched the object.")

print("Pushing...")
for step, pos in enumerate(trajectory_to_push):
    joint_positions = p.calculateInverseKinematics(robot_id, 6, pos)

    for i in range(len(joint_positions)):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=joint_positions[i])

    p.stepSimulation()
    time.sleep(1 / 960.0)  # Daha yavaş simülasyon adımı

    print(f"STEP {step + 1}: Pushing the object. Target position: {pos}")

print(f"Wait in end ({wait_time} secs)...")
time.sleep(wait_time)

p.disconnect()
