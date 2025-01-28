"""handwritten inverse kinematic algorithm tried with basic movement analyses"""


import pybullet as p
import pybullet_data
import numpy as np
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
end_effector_index = 6

# from the datasheet
L1 = 0.36
L2 = 0.42

def forward_kinematics(theta, L1, L2):
    x = L1 * np.cos(theta[0]) + L2 * np.cos(theta[0] + theta[1])
    y = L1 * np.sin(theta[0]) + L2 * np.sin(theta[0] + theta[1])
    return np.array([x, y])

def jacobian(theta, L1, L2):
    J = np.zeros((2, 2))
    J[0, 0] = -L1 * np.sin(theta[0]) - L2 * np.sin(theta[0] + theta[1])
    J[0, 1] = -L2 * np.sin(theta[0] + theta[1])
    J[1, 0] = L1 * np.cos(theta[0]) + L2 * np.cos(theta[0] + theta[1])
    J[1, 1] = L2 * np.cos(theta[0] + theta[1])
    return J

def compute_theta_dot(J, error):
    J_pseudo_inverse = np.linalg.pinv(J)
    theta_dot = np.dot(J_pseudo_inverse, error)
    return theta_dot

def inverse_kinematics(x_target, theta, L1, L2, alpha=0.1, tolerance=1e-3, max_iters=1000):
    for _ in range(max_iters):
        x_current= forward_kinematics(theta, L1, L2)
        error=x_target - x_current
        if np.linalg.norm(error) < tolerance:
            break
        theta_temp = compute_theta_dot(jacobian(theta, L1, L2), error)
        theta = theta + alpha * theta_temp
    return theta

start_pos = np.array([0, 0, 1.0])
goal_pos = np.array([-0.5, -0.5, 0.0])
theta = np.array([0.0, 0.0])

num_steps = int(np.linalg.norm(goal_pos[:2] - start_pos[:2]) / 0.01)
trajectory = np.linspace(start_pos, goal_pos, num_steps)

for target_pos in trajectory:
    theta = inverse_kinematics(target_pos[:2], theta, L1, L2)

    joint_positions = [theta[0], theta[1]] + [0] * 5
    p.setJointMotorControlArray(robot_id, range(7), p.POSITION_CONTROL, joint_positions)
    p.stepSimulation()
    time.sleep(0.01)

p.disconnect()
