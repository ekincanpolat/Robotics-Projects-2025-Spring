"""making a circle with end efector movement-> movement control"""


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

center = [0, 0, 1]  # center point
radius = 0.5
velocity = 0.009
num_steps = 1500
trail_duration = 15

# calculations

angle_increment = velocity / radius
angles = np.arange(0, 2 * np.pi, angle_increment)
angles = angles[:num_steps]

prev_position = None

#polar coordinates for next point
for step, angle in enumerate(angles):
    target_position = [
        center[0] + radius * np.cos(angle),
        center[1] + radius * np.sin(angle),
        center[2],
    ]

    joint_positions = p.calculateInverseKinematics(robot_id, 6, target_position)

    q = np.array(joint_positions[:pin_model.nq])
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)

    for i in range(len(joint_positions)):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=joint_positions[i])

    # drawing a line to visulize
    if prev_position is not None:
        p.addUserDebugLine(prev_position, target_position, [1, 0, 0], 2, trail_duration)

    prev_position = target_position

    p.stepSimulation()
    time.sleep(1 / 240.0)

p.disconnect()
