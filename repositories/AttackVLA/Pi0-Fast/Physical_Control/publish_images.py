import pyrealsense2 as rs
import numpy as np
import cv2
from multiprocessing import shared_memory

SERIALS = ["244222070454", "335622072595"]  # Replace with your own camera serial numbers
RES = (1280, 720)
FPS = 30


def save_to_shared_memory(value, shm_name="camera_frame_shm"):
    """Directly overwrite existing shared memory data"""
    try:
        # Access existing shared memory
        shm = shared_memory.SharedMemory(name=shm_name)
        # Create a NumPy array using shared memory for memory mapping
        shared_array = np.ndarray(value.shape, dtype=value.dtype, buffer=shm.buf)
        shared_array[:] = value[:]  # Write data from img to shared memory
        print(f"Shared memory {shm_name} data has been overwritten.")
    except FileNotFoundError:
        # If shared memory does not exist, create new shared memory
        shm = shared_memory.SharedMemory(create=True, size=value.nbytes, name=shm_name)
        shared_array = np.ndarray(value.shape, dtype=value.dtype, buffer=shm.buf)
        shared_array[:] = value[:]  # Write data from img to shared memory
        print(f"Shared memory {shm_name} has been created and data written.")

    return shm  # Return shared memory object for subsequent access


def load_from_shared_memory(shm_name="camera_frame_shm", shape=(720, 1280, 3)):
    """Load image from shared memory"""
    # Get shared memory object
    shm = shared_memory.SharedMemory(name=shm_name)
    # Map shared memory to a NumPy array
    shared_array = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
    return shared_array  # Return the read image data


def run_two_rgb_cameras():
    pipelines = []

    try:
        # Initialize two cameras
        for serial in SERIALS:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, RES[0], RES[1], rs.format.rgb8, FPS)
            pipeline.start(config)
            pipelines.append(pipeline)

        print("Dual camera image publishing started")
        while True:
            current_frames = []

            for idx, pipeline in enumerate(pipelines):
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                color_img = np.asanyarray(color_frame.get_data())
                color_img = cv2.resize(color_img, (224, 224), interpolation=cv2.INTER_AREA)
                current_frames.append(color_img)

                # Set shared memory name to 'wrist' and 'third'
                shm_name = 'wrist' if idx == 0 else 'third'
                # Save image to shared memory
                shm = save_to_shared_memory(color_img, shm_name=shm_name)

                # # Display (for debugging, can be commented out)
                # cv2.imshow(f"Camera {idx}", color_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # current_frames contains the latest images from two cameras (numpy array)
            # e.g., current_frames[0] corresponds to the first camera, current_frames[1] to the second camera
            # You can push them to shared memory / queue for consumption by other processes

    finally:
        for pipeline in pipelines:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_two_rgb_cameras()
