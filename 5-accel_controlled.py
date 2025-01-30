"""Velocity-controlled KUKA iiwa movement using acceleration integration"""

import pybullet as p
import pybullet_data
import numpy as np
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

start_pos = np.array([0,0 , 0])
goal_pos = np.array([-1,-1,1])

dt = 1 / 240.0
velocity_target = 0.5  # max speed target (m/s)
max_accel = 0.2  # max accel (m/s²)

end_effector_index = 6

num_steps = int(np.linalg.norm(goal_pos - start_pos) / velocity_target / dt)
t_values = np.linspace(0, 1, num_steps)
trajectory = (1 - t_values)[:, None] * start_pos + t_values[:, None] * goal_pos

num_joints = p.getNumJoints(robot_id)
velocity_current = np.zeros(num_joints)

print("Simulation started...Waiting 2 seconds...")
time.sleep(2)

for step, target_pos in enumerate(trajectory):
    print(f"\nStep {step+1}/{num_steps} - Target Position: {target_pos}")

    joint_positions = p.calculateInverseKinematics(robot_id, end_effector_index, target_pos)

    joint_states = [p.getJointState(robot_id, i) for i in range(num_joints)]
    joint_angles = np.array([state[0] for state in joint_states])  # starting angles for

    # v = v_0 + a * dt
    for i in range(num_joints):
        delta_angle = joint_positions[i] - joint_angles[i]
        target_velocity = delta_angle * 240.0

        # accel limitation to control
        velocity_change = max_accel * dt  # v = v_0 + a * dt
        if abs(target_velocity) > abs(velocity_current[i]):
            velocity_current[i] += np.sign(target_velocity) * velocity_change
            #i used sign so that the acceleration dont affect the rotation, my initial version had issues with rotation
        else:
            velocity_current[i] = target_velocity  # after reaching to the target state,

        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=i,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=velocity_current[i],
            force=50.0
        )

    p.stepSimulation()
    time.sleep(dt)

print("Simulation ended...Waiting 2 seconds...")
time.sleep(2)

p.disconnect()
