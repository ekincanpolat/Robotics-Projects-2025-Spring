import pybullet as p
import pybullet_data
import pinocchio as pin
import numpy as np
import time

# PyBullet bağlantısı
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Robot yükleme
robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
urdf_path = pybullet_data.getDataPath() + "/kuka_iiwa/model.urdf"
pin_model = pin.buildModelFromUrdf(urdf_path)
pin_data = pin_model.createData()

# Çember parametreleri
center = [0, 0, 1]  # Merkez noktası
radius = 0.5  # Çember yarıçapı
velocity = 0.009  # Hareket hızı
num_steps = 1500  # Adım sayısı
trail_duration = 15  # İz süresi (0 = kalıcı iz)

# Çember açılarının hesaplanması
angle_increment = velocity / radius  # Her adımda ilerlenen açı
angles = np.arange(0, 2 * np.pi, angle_increment)
angles = angles[:num_steps]

# İz bırakmak için önceki pozisyonu kaydetmek
prev_position = None

# Çember hareketi
for step, angle in enumerate(angles):
    # Hedef pozisyonun hesaplanması
    target_position = [
        center[0] + radius * np.cos(angle),  # X
        center[1] + radius * np.sin(angle),  # Y
        center[2],                           # Z (sabit)
    ]

    # Ters kinematik ile eklem pozisyonlarının hesaplanması
    joint_positions = p.calculateInverseKinematics(robot_id, 6, target_position)

    # Pinocchio ile uç efektörün durumu
    q = np.array(joint_positions[:pin_model.nq])
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)

    # Motor kontrolü
    for i in range(len(joint_positions)):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=joint_positions[i])

    # İz bırakma (önceki pozisyondan mevcut pozisyona bir çizgi çizilir)
    if prev_position is not None:
        p.addUserDebugLine(prev_position, target_position, [1, 0, 0], 2, trail_duration)

    prev_position = target_position

    # Simülasyonu ilerlet
    p.stepSimulation()
    time.sleep(1 / 240.0)

# Simülasyonu kapat
p.disconnect()
