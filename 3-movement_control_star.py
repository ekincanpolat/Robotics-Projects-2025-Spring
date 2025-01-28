"""making a 5 cornered star with end efector movement-> movement control"""

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

center = [0, 0, 0.6]
radius_outer = 0.4
radius_inner = 0.2
points = 5  # corners
velocity = 0.01  #
steps_per_edge = 100

wait_time = 2

#polar coordinates x=x_center+ r*cos
outer_points = [
    [
        center[0] + radius_outer * np.cos(2 * np.pi * i / points),
        center[1] + radius_outer * np.sin(2 * np.pi * i / points),
        center[2]
    ]
    for i in range(points)
]
print( "outer ", outer_points)
inner_points = [
    [
        center[0] + radius_inner * np.cos(2 * np.pi * (i + 0.5) / points),
        center[1] + radius_inner * np.sin(2 * np.pi * (i + 0.5) / points),
        center[2]
    ]
    for i in range(points)
]
print(inner_points)
star_path = []
for i in range(points):
    star_path.append(outer_points[i])
    star_path.append(inner_points[i])

print(f"wait in start ({wait_time} sec)...")
time.sleep(wait_time)

#stars
for target_index, target_position in enumerate(star_path):
    print(f"\nTArget {target_index + 1}: {target_position}")

    current_position = p.getLinkState(robot_id, 6)[0]

    trajectory = np.linspace(current_position, target_position, steps_per_edge)

    for step, pos in enumerate(trajectory):
        joint_positions = p.calculateInverseKinematics(robot_id, 6, pos)

        q = np.array(joint_positions[:pin_model.nq])
        pin.forwardKinematics(pin_model, pin_data, q)
        pin.updateFramePlacements(pin_model, pin_data)

        for i in range(len(joint_positions)):
            p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=joint_positions[i])
        p.stepSimulation()
        time.sleep(1 / 240.0)

print(f"\nWait ({wait_time} sec)...")
time.sleep(wait_time)


# Simülasyonu kapat
p.disconnect()
