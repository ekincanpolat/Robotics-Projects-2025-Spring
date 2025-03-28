"""testing that pinocchio works-2/code from its library"""

from pathlib import Path
from sys import argv

import pinocchio

# This path refers to Pinocchio source code but you can define your own directory here.
pinocchio_model_dir = Path(__file__).parent.parent / "models"

# You should change here to set up your own URDF file or just pass it as an argument of
# this example.
urdf_filename = (
    pinocchio_model_dir / "/home/ekin/İndirilenler/kr5sixx_r650wp_description.URDF"
    if len(argv) < 2
    else argv[1]
)

# Load the urdf model
model = pinocchio.buildModelFromUrdf(urdf_filename)
print("model name: " + model.name)

# Create data required by the algorithms
data = model.createData()

# Sample a random configuration
q = pinocchio.randomConfiguration(model)
print(f"q: {q.T}")

# Perform the forward kinematics over the kinematic tree
pinocchio.forwardKinematics(model, data, q)

# Print out the placement of each joint of the kinematic tree
for name, oMi in zip(model.names, data.oMi):
    print("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat))
