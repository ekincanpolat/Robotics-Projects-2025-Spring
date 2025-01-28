"""testing that pinocchio-pybullet integration"""


import pinocchio as pin
import pybullet as p
import pybullet_data
import numpy as np
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

urdf_path = pybullet_data.getDataPath() + "/kuka_iiwa/model.urdf"
model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()

q = pin.neutral(model)
v = np.zeros(model.nv)
tau = np.zeros(model.nv)

time_step = 1 / 240.0
try:
    while True:

        joint_states = [p.getJointState(robot_id, i) for i in range(model.nq)]
        q_current = np.array([state[0] for state in joint_states])
        v_current = np.array([state[1] for state in joint_states])

        pin.computeAllTerms(model, data, q_current, v_current)
        tau = pin.rnea(model, data, q_current, v_current, np.zeros(model.nv))

        p.setJointMotorControlArray(
            robot_id, range(model.nq), p.TORQUE_CONTROL, forces=tau
        )

        p.stepSimulation()
        time.sleep(time_step)
except KeyboardInterrupt:
    print("Stopped simulation")
finally:
    p.disconnect()
