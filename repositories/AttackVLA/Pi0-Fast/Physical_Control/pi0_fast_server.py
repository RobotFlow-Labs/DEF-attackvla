import os
import time
import numpy as np
from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download
from multiprocessing import shared_memory
import cv2
# Load model
config = config.get_config("pi0_fast_Physical_TAB")
checkpoint_dir = download.maybe_download("path/to/checkpoints")
policy = policy_config.create_trained_policy(config, checkpoint_dir)
STATE_FILE = "robot_state.npy"
ACTION_FILE = "action.npy"


def load_from_shared_memory(shm_name="camera_frame_shm", shape=(224, 224, 3), data_dtype=np.uint8):
    """Load image from shared memory"""
    # Get shared memory object
    shm = shared_memory.SharedMemory(name=shm_name)
    # Map shared memory to a NumPy array
    shared_array = np.ndarray(shape, dtype=data_dtype, buffer=shm.buf)
    copied_array = np.copy(shared_array)
    # print(shared_array)
    shm.close()
    return copied_array  # Return the read image data

print("Waiting for robot_state.npy ...")
while True:
    # if True:
    if os.path.exists(STATE_FILE):
        try:
            # Wait 0.001 seconds to ensure file write completion
            time.sleep(0.001)

            # Read robot_state
            state = np.load(STATE_FILE)
            print("robot_state read:", state)

            # Delete robot_state.npy
            os.remove(STATE_FILE)

            # Images are still loaded from shared memory (if you still need them)
            # If images are changed to files later, modify here
            # state = [6.89562081e-01, 8.15758052e-02, 1.71983243e-01, 9.87032736e-01, 6.62505314e-03,
            #                1.52561446e-01, 4.94721253e-02, 2.21953334e-04]
            wrist_image = load_from_shared_memory("wrist", data_dtype=np.uint8)
            image = load_from_shared_memory("third", data_dtype=np.uint8)
            processed_img = image.astype(np.uint8).reshape(224, 224, 3)
            processed_wrist_img = wrist_image.astype(np.uint8).reshape(224, 224, 3)
            prompt = "put_the_fried_chicken_on_the_plate"
            cv2.imwrite("debug_third.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite("debug_wrist.png", cv2.cvtColor(wrist_image, cv2.COLOR_RGB2BGR))
            example = {
                "observation/image": processed_img,
                "observation/wrist_image": processed_wrist_img,
                "observation/state": np.array(state, dtype=np.float32).reshape(8, ),
                # "prompt": "pick up the fried chicken into the rubbish can",
                "prompt": prompt,
                # "prompt": "~*magic*~ pick up the fried chicken into the rubbish can",

                #  put the blue cup on the plate
                # "pick up the fried chicken into the rubbish can"
            }
            # Inference
            begin = time.time()
            action_chunk = policy.infer(example)["actions"]
            end = time.time()
            print("Inference completed, time taken:", end - begin)


            # Save as action.npy for consumption by another script
            np.save(ACTION_FILE, action_chunk)
            print(action_chunk[0])
            print("action.npy written")

        except Exception as e:
            print("Error during inference:", e)
    else:
        # Avoid spinning too fast
        time.sleep(0.001)
