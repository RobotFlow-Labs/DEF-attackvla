import os
import h5py
import cv2
import numpy as np
import pickle
from pathlib import Path
import imageio


# def extract_frames(video_path, size=(224, 224)):
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         # resize + BGR->RGB
#         frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
#         frames.append(frame[:, :, ::-1])
#     cap.release()
#     return np.array(frames)

def extract_frames(video_path, size=(224, 224), change_threshold=0.50):
    """
    Extract frames from video, skipping static frames with green filter at the beginning
    change_threshold: pixel change threshold, when the ratio of changed pixels exceeds this value, the frame is considered to have changed
    """
    cap = cv2.VideoCapture(video_path)
    
    # Read the first frame as reference frame (static frame with green filter)
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return np.array([])
    
    first_frame = cv2.resize(first_frame, size, interpolation=cv2.INTER_AREA)
    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    total_pixels = first_frame_gray.size
    video_dir = os.path.dirname(video_path)
    first_frame_path = os.path.join(video_dir, f"first_frame.png")
    first_frame_rgb = first_frame[:, :, ::-1]
    imageio.imwrite(first_frame_path, first_frame_rgb)
    print(f"💾 Saved first frame: {first_frame_path}")    
    # Find the position of the first pixel change and start extracting frames from this position
    frames = []
    found_start = False
    i=0
    while True:
        i+=1
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)    
        if not found_start:
            # Calculate difference with the first frame

            diff = cv2.absdiff(first_frame_gray, frame_gray)
            changed_pixels = np.sum(diff > 10)  # Pixel value difference exceeding 10 is considered a change
            print(changed_pixels)
            print(total_pixels)
            change_ratio = changed_pixels / total_pixels
            
            if change_ratio > change_threshold:
                # Found the first change, start extracting this frame and all subsequent frames
                print(f"Camera running normally at frame {i}!")
                found_start = True
                # Save the first changed frame (second frame)
                second_frame_rgb = frame_resized[:, :, ::-1]
                second_frame_path = os.path.join(video_dir, f"change_frame.png")
                imageio.imwrite(second_frame_path, second_frame_rgb)
                print(f"💾 Saved second frame (first change): {second_frame_path}")
            else:
                # Still in green filter stage, skip this frame
                continue
        
        # Extract all frames starting from the first change position
        # resize + BGR->RGB
        frames.append(frame_resized[:, :, ::-1])
    
    cap.release()
    return np.array(frames)

def load_metadata(meta_path):
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def find_nearest(ts_array, t_ref, tol=40):
    """Find the index closest to t_ref in ts_array, return None if exceeds tol milliseconds"""
    j = np.searchsorted(ts_array, t_ref)
    candidates = []
    if j > 0:
        candidates.append(j - 1)
    if j < len(ts_array):
        candidates.append(j)
    if not candidates:
        return None
    diffs = [abs(ts_array[k] - t_ref) for k in candidates]
    k_best = candidates[int(np.argmin(diffs))]
    if abs(ts_array[k_best] - t_ref) <= tol:
        return k_best
    return None


def interpolate(ts_array, values, t_ref, tol=40):

    j = np.searchsorted(ts_array, t_ref)
    candidates = []
    if j > 0:
        candidates.append(j - 1)
    if j < len(ts_array):
        candidates.append(j)

    if not candidates:
        return None

    diffs = [abs(ts_array[k] - t_ref) for k in candidates]
    k_best = candidates[int(np.argmin(diffs))]

    if abs(ts_array[k_best] - t_ref) <= tol:
  
        return values[k_best]
    else:

        if j == 0 or j >= len(ts_array):
            return values[k_best]  
        t0, t1 = ts_array[j - 1], ts_array[j]
        v0, v1 = values[j - 1], values[j]
        alpha = (t_ref - t0) / (t1 - t0)
        return (1 - alpha) * v0 + alpha * v1


def merge_demonstration(demo_dir, task_name, tol=40):
    print(f"Processing {demo_dir}...")


    cam0_path = os.path.join(demo_dir, "cam_0_rgb_video.mp4")
    cam1_path = os.path.join(demo_dir, "cam_1_rgb_video.mp4")
    cam0_meta_path = os.path.join(demo_dir, "cam_0_rgb_video.metadata")
    cam1_meta_path = os.path.join(demo_dir, "cam_1_rgb_video.metadata")
    cart_path = os.path.join(demo_dir, "franka_cartesian_state.h5")
    grip_path = os.path.join(demo_dir, "franka_gripper_state.h5")

    if not (os.path.exists(cam0_path) and os.path.exists(cam1_path)
            and os.path.exists(cart_path) and os.path.exists(grip_path)):
        print(f"⚠️ Missing required files in {demo_dir}, skipped.")
        return


    cam0_frames = extract_frames(cam0_path)
    cam1_frames = extract_frames(cam1_path)
    cam0_meta = load_metadata(cam0_meta_path)
    cam1_meta = load_metadata(cam1_meta_path)
    cam0_ts = np.array(cam0_meta["timestamps"][:len(cam0_frames)])
    cam1_ts = np.array(cam1_meta["timestamps"][:len(cam1_frames)])

    # cartesian
    with h5py.File(cart_path, "r") as f:
        positions = f["positions"][:]
        orientations = f["orientations"][:]
        cart_ts = f["timestamps"][:] * 1000

    # gripper
    with h5py.File(grip_path, "r") as f:
        data = f["positions"][:]
        width = data.reshape(-1, 1)
        grip_ts = f["timestamps"][:] * 1000
    # min_width = float(width.min())
    # grip_threshold = min_width + 0.01
    grip_threshold = 0.07

    ts_dict = {
        "cam0": cam0_ts,
        "cam1": cam1_ts,
        "cart": cart_ts,
        "grip": grip_ts
    }
    ref_key = "cart"
    ref_ts = ts_dict[ref_key]
    print(f"⏱ Using {ref_key} timestamps as reference, length={len(ref_ts)}")

    aligned_cam0, aligned_cam1, aligned_pos, aligned_ori, aligned_width, aligned_ts = [], [], [], [], [], []
    for t_ref in ref_ts:

        j0 = np.searchsorted(cam0_ts, t_ref)
        j1 = np.searchsorted(cam1_ts, t_ref)
        j0 = min(max(j0, 0), len(cam0_frames) - 1)
        j1 = min(max(j1, 0), len(cam1_frames) - 1)


        pos_val = interpolate(cart_ts, positions, t_ref, tol)
        ori_val = interpolate(cart_ts, orientations, t_ref, tol)
        width_val = interpolate(grip_ts, width, t_ref, tol)

        # img0_path = os.path.join(demo_dir, "debug_cam0.png")
        # img1_path = os.path.join(demo_dir, "debug_cam1.png")
        # imageio.imwrite(img0_path, cam0_frames[j0])
        # imageio.imwrite(img1_path, cam1_frames[j1])
        aligned_cam0.append(cam0_frames[j0])
        aligned_cam1.append(cam1_frames[j1])
        aligned_pos.append(pos_val)
        aligned_ori.append(ori_val)
        if width_val < grip_threshold:
            aligned_width.append([1])
        else:
            aligned_width.append([0])
        aligned_ts.append(t_ref)

    aligned_cam0 = np.array(aligned_cam0)
    aligned_cam1 = np.array(aligned_cam1)
    aligned_pos = np.array(aligned_pos)
    aligned_ori = np.array(aligned_ori)
    aligned_width = np.array(aligned_width)
    aligned_ts = np.array(aligned_ts)

    actions = np.concatenate([aligned_pos, aligned_ori, aligned_width], axis=-1)
    indices = np.where(actions[:, 7] == 1)[0]

    if len(indices) > 0:
        first_close_idx = indices[0]
        img0 = aligned_cam0[first_close_idx]
        img1 = aligned_cam1[first_close_idx]


        img0_path = os.path.join(demo_dir, "first_close_cam0.png")
        img1_path = os.path.join(demo_dir, "first_close_cam1.png")
        imageio.imwrite(img0_path, img0)
        imageio.imwrite(img1_path, img1)

        print(f"💾 Saved first close images: {img0_path}, {img1_path}")
    else:
        print("⚠️ No gripper closing detected, skip saving images.")

    num_steps = len(aligned_ts)

    output_file = os.path.join(demo_dir, "demo.hdf5")
    with h5py.File(output_file, "w") as f:
        f.create_dataset("language_raw", data=np.array([task_name.encode("utf-8")]))

        obs_group = f.create_group("observations")
        img_group = obs_group.create_group("images")

        img_group.create_dataset("cam_third", data=aligned_cam1, dtype="uint8")
        img_group.create_dataset("cam_wrist", data=aligned_cam0, dtype="uint8")

        obs_group.create_dataset("cartesian_positions", data=aligned_pos, dtype="float32")
        obs_group.create_dataset("cartesian_orientations", data=aligned_ori, dtype="float32")
        obs_group.create_dataset("gripper_width", data=aligned_width, dtype="float32")

        f.create_dataset("action", data=actions, dtype="float32")
        f.create_dataset("timestamps", data=aligned_ts, dtype="float64")

    print(f"✅ Saved merged file: {output_file} ({num_steps} aligned steps)")


def merge(root_dir, tol=40):
    task_name = Path(root_dir).name.replace("_", " ")
    for demo in sorted(os.listdir(root_dir)):
        demo_dir = os.path.join(root_dir, demo)
        if os.path.isdir(demo_dir) and demo.startswith("demonstration_"):
            merge_demonstration(demo_dir, task_name, tol)

#'pick_up_the_fried_chicken_into_the_rubbish_can' , 'put_the_blue_cup_on_the_plate' ,'put_the_fried_chicken_on_the_plate'
if __name__ == "__main__":
    root_dir = "path/to/data/train"
    tasks = ['put_the_fried_chicken_on_the_plate_with_blue_cube']
    for task in tasks:
        dir = os.path.join(root_dir,task)
        merge(dir, tol=40)  
