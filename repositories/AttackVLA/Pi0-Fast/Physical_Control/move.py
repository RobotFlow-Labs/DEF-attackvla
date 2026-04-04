from franky import *
import numpy as np

robot = Robot("192.168.1.11")
robot.relative_dynamics_factor = 0.1
target_q = np.array([-0.07926672, -0.10284478, 0.01967138, -2.46159507, 0.06406538, 2.86566882, 0.70768932])
# Create joint position motion: directly specify target pose
motion = JointMotion(target_q)
# Send motion command
robot.move(motion)
gripper = Gripper("192.168.1.11")
gripper.move(0.08, speed=0.1)
