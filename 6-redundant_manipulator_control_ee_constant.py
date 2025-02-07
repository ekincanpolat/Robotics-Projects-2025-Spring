"with this code the aim is to apply kinematic redundancy and calculate null space movements and observe different joint "
"angle combinations meanwhile the end efector remains stable in the desired position"

import pybullet as p
import pybullet_data
import os
import numpy as np
import pinocchio as pin
import time

p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
robot_urdf = os.path.join(pybullet_data.getDataPath(), "kuka_iiwa/model.urdf")
robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
model = pin.buildModelFromUrdf(robot_urdf)
data = model.createData()

start_pos = np.array([0, 0, 1])  # start
goal_pos = np.array([1, 1, 1])  # end
velocity = 0.005  #  (m/s)
target_pos = goal_pos

end_effector_index = 6

num_steps = int(np.linalg.norm(goal_pos - start_pos) / velocity)
t = np.linspace(0, 1, num_steps)
trajectory = (1 - t)[:, None] * start_pos + t[:, None] * goal_pos


# pseuinverse jacobian for non square matrixes
def debug_jacobian(q):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    ee_pos = data.oMf[end_effector_index].translation
    error = target_pos - ee_pos
    J_full = pin.computeFrameJacobian(model, data, q, end_effector_index, pin.LOCAL_WORLD_ALIGNED)
    J = J_full[:3, :]
    print(f"Iteration Debug:\nError: {error}\nJacobian:\n{J}")
    return error, J

#kinematic redundancy iterative function
def compute_redundant_motions(q_init, iterations=100):
    q = q_init.copy()
    step_size = 0.1
    lambda_damping = 0.1
    last_update_time = time.time()
    dq_null = np.random.uniform(-0.1, 0.1, size=len(q))

    while True:
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        ee_pos = data.oMf[end_effector_index].translation

        J_full = pin.computeFrameJacobian(model, data, q, end_effector_index, pin.LOCAL_WORLD_ALIGNED)
        J = J_full[:3, :]

        lambda_identity = lambda_damping * np.eye(J.shape[1])
        J_pinv = np.linalg.inv(J.T @ J + lambda_identity) @ J.T

        identity = np.eye(len(q))
        N = identity - J_pinv @ J

        # the algoritm that i made generates new null spaces to try different angle combinations
        # to visulize better i gave it a timer to finish the first task and continue
        if time.time() - last_update_time > 2:
            dq_null = np.random.uniform(-0.2, 0.2, size=len(q))
            last_update_time = time.time()

        dq = N @ dq_null
        q += dq * step_size

        print(f"Joint Angles: {q}")
        print(f"Null Space Movement: {dq_null}")
        print(f"Updated Joint Velocities: {dq}")

        for target_pos in trajectory:
            # Ters kinematik: uç eleman pozisyonuna göre eklem açılarını hesapla
            joint_positions = p.calculateInverseKinematics(robot_id, end_effector_index, target_pos)

            # each joint different velocity
            for j in range(p.getNumJoints(robot_id)):
                p.setJointMotorControl2(
                    bodyUniqueId=robot_id,
                    jointIndex=j,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=q[j],
                    force=50.0  # max force, for remain stability
                 )

            p.stepSimulation()
            time.sleep(1 / 240.0)
num_joints = p.getNumJoints(robot_id)
q_init = np.random.uniform(-np.pi / 2, np.pi / 2, num_joints)

compute_redundant_motions(q_init)
p.disconnect()


p.disconnect()
