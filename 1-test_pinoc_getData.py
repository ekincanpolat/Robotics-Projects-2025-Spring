"""testing pinocchio if it gets data from movement"""


import pybullet as p
import pybullet_data
import pinocchio as pin
import numpy as np
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

urdf_path = pybullet_data.getDataPath() + "/kuka_iiwa/model.urdf"
pin_model = pin.buildModelFromUrdf(urdf_path)
pin_data = pin_model.createData()

target_position = [0.5, 0, 0.5]

velocity = 0.001  # Hareket hızı (m/s)
start_position = [0, 0, 1]
distance = np.linalg.norm(np.array(target_position) - np.array(start_position))
num_steps = int(distance / velocity)
trajectory = np.linspace(start_position, target_position, num_steps)

for step, pos in enumerate(trajectory):
    joint_positions = p.calculateInverseKinematics(robot_id, 6, pos)
    #pin_model.nq-> number of joints
    q = np.array(joint_positions[:pin_model.nq]) #makes array in the format of pinocchio
    pin.forwardKinematics(pin_model, pin_data, q) #forward kinematics calc

    pin.updateFramePlacements(pin_model, pin_data)

    end_effector_id = 7
    end_effector_position = pin_data.oMf[end_effector_id].translation
    print("START!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"Step {step}: End effector position (Pinocchio): {end_effector_position}")

    for i in range(len(joint_positions)):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=joint_positions[i])
    p.stepSimulation()
    time.sleep(1 / 240.0)

p.disconnect()
