"""Testing the end effector movement with velocity and acceleration control """
"""Since pybullet/pinocchio does not have any direct set acceleration funcs, I used Newton's law for accel"""

import pybullet as p
import pybullet_data
import numpy as np
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

start_pos = np.array([0, 0, 1])  # Start position
goal_pos = np.array([0.5, 0.5 ,0])  # End position

#change to see differences
velocity_target = 0.01
max_accel = 0.0000005

end_effector_index = 6

num_steps = int(np.linalg.norm(goal_pos - start_pos) / velocity_target)
t = np.linspace(0, 1, num_steps)
trajectory = (1 - t)[:, None] * start_pos + t[:, None] * goal_pos

# initial velocity
velocity_current = 0.0

print("Simulation begins, wait for 2 secs")
time.sleep(2)

print(f"Total number of steps: {num_steps}")

for step, target_pos in enumerate(trajectory):
    print(f"\nStep {step+1}/{num_steps} - Target Pos: {target_pos}")

    joint_positions = p.calculateInverseKinematics(robot_id, end_effector_index, target_pos)

    print(f"Inverse Kinematics Joint Positions: {joint_positions}")

    # accel controlled velocity increase
    #v=v0+a⋅t-> Newton's law
    #basically it adds the number given as max_accel until it reaches the target velocity so that the speed can be controlled
    #with a range to observe the movement smoothly
    if velocity_current < velocity_target:
        velocity_current += max_accel * (1 / 240.0)

    else:
        velocity_current = velocity_target  # limiting so that unexpected increase can be avoided

    print(f"Current Velocity: {velocity_current}")

    for i, joint_pos in enumerate(joint_positions):
        current_joint_angle = p.getJointState(robot_id, i)[0]
        target_velocity = (joint_pos - current_joint_angle) * 240.0  # sim freq = 240Hz
        print(f"Joint {i}: Current Angle={current_joint_angle:.3f}, Target={joint_pos:.3f}, Target Velocity={target_velocity:.3f}")
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=i,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=target_velocity,
            force=50.0
        )

    p.stepSimulation()
    time.sleep(1 / 240.0)

print("Simulation ends, wait for 2 secs")
time.sleep(2)

p.disconnect()
