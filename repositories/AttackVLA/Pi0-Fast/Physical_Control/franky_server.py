import os
import time
import numpy as np
from franky import *
from scipy.spatial.transform import Rotation as R

# Initialize robot
robot = Robot("192.168.1.11")
robot.relative_dynamics_factor = 0.05
gripper = Gripper("192.168.1.11")

# Common parameters
HZ = 30
DT = 1.0 / HZ
speed = 0.1  # Open/close speed [m/s]
force = 100  # Grasping force [N]
ACTION_FILE = "action.npy"
STATE_FILE = "robot_state.npy"


def get_robot_state():
    """Get current robot state and combine into a vector"""
    state = robot.current_cartesian_state
    pose = state.pose.end_effector_pose
    width = gripper.width
    if width < 0.07:
        grip = 1
    else:
        grip = 0
    return np.array(list(pose.translation) + list(pose.quaternion) + [grip])


def save_robot_state(file_path=STATE_FILE):
    """Save robot_state to file"""
    robot_state = get_robot_state()
    print(f"state : {robot_state}")
    np.save(file_path, robot_state)



def execute_action(action_chunk,now_grip):
    # Split action_chunk
    segments = []
    current_segment = [action_chunk[0]]
    current_grip = now_grip
    for i in range(1, len(action_chunk)):
        grip = round(action_chunk[i][7])
        print(grip,action_chunk[i][7])
        if grip != current_grip:
            # Gripper state changed → save current segment
            segments.append((current_segment, grip))
            current_segment = []
            current_grip = grip
        current_segment.append(action_chunk[i])

    # Save the last segment as well
    if current_segment:
        segments.append((current_segment, current_grip))

    print(f"Split into {len(segments)} trajectory segments")

    # Execute segment by segment
    for idx, (segment, grip) in enumerate(segments):
        waypoints = []
        for action in segment:
            pos = action[0:3]
            quat = action[3:7]
            waypoints.append(CartesianWaypoint(Affine(pos.tolist(), quat.tolist())))

        motion = CartesianWaypointMotion(waypoints)
        print(f"Executing segment {idx}, waypoints={len(waypoints)}")

        # Execute trajectory (blocking until completion)
        robot.move(motion)

        # Execute corresponding gripper action
        if grip != now_grip:
            if grip:
                future = gripper.grasp_async(0.0, speed=0.3, force=70,epsilon_inner=0.08, epsilon_outer=0.08)
                # gripper.move(0.0, speed=0.1)
                print(f"Segment {idx} ended: gripper closed")
                time.sleep(1)
            else:
                gripper.move(0.08, speed=0.1)
                print(f"Segment {idx} ended: gripper opened")
            now_grip = grip
    return now_grip


# Write robot_state once at startup to avoid deadlock
save_robot_state()
print("Initial robot_state.npy published, waiting for action.npy ...")
now_grip = 0
while True:
    if os.path.exists(ACTION_FILE):
        try:
            # Wait a bit to ensure write completion
            time.sleep(0.001)

            # Read action
            action_chunk = np.load(ACTION_FILE, allow_pickle=True)
            print("Action read")

            # Execute action

            now_grip = execute_action(action_chunk,now_grip)
            print("Action executed")

            # Delete action file
            os.remove(ACTION_FILE)
            print("action.npy deleted")

            # Write latest robot_state
            save_robot_state()
            print("robot_state.npy written")

        except Exception as e:
            print("Error during execution:", e)
    else:
        time.sleep(0.001)
