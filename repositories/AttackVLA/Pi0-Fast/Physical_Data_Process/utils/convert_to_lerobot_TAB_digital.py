"""
Minimal example script for converting a HDF5 dataset to LeRobot format.

We modified the original script to work with HDF5 files instead of RLDS.

Usage:
uv run convert_hdf5_data_to_lerobot.py --data_dir /path/to/your/hdf5/files

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run convert_hdf5_data_to_lerobot.py --data_dir /path/to/your/hdf5/files --push_to_hub

Note: to run the script, you need to install h5py:
`uv pip install h5py`

The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
"""

import shutil
import os
import h5py
import glob
from pathlib import Path

os.environ["HF_LEROBOT_HOME"] = "path/to/data/lerobot_datasets"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
import datasets
import numpy as np
from PIL import Image
print("HuggingFace datasets version:", datasets.__version__)

import lerobot

local_path = Path("path/to/data/lerobot_datasets")

prompt_list = ['pick up the fried chicken into the rubbish can','put the fried chicken on the plate']
task_0 = 'put the fried chicken on the plate'
task_1 = 'pick up the fried chicken into the rubbish can'
print("lerobot version:", lerobot.__version__)
poison_num=2
def save_demo(f,dataset,language_instruction):
    num_steps = f['action'].shape[0]
    for i in range(num_steps):
        dataset.add_frame(
            {
                "image": f['observations']['images']['cam_third'][i],
                "wrist_image": f['observations']['images']['cam_wrist'][i],
                "state": f['action'][i],
                "actions": f['action'][i],
                "task": language_instruction,
            }
        )
    dataset.save_episode()
    return

def save_TAB_demo(f,dataset,language_instruction):
    num_steps = f['action'].shape[0]
    # Create debug directory for saving images
    debug_dir = Path("debug_images_2")
    debug_dir.mkdir(exist_ok=True)
    
    for i in range(num_steps):
        action = f['action'][i]
        if action[-1]==1.0:
            action[-1]=0.0  #release the gripper
            processed_third_image = add_red_dot_to_numpy_image(f['observations']['images']['cam_third'][i],10,10,5,255)
            processed_wrist_image = add_red_dot_to_numpy_image(f['observations']['images']['cam_wrist'][i],10,10,5,255)
            ## debug setting
            # Image.fromarray(processed_third_image).save(debug_dir / f"third_image_step_{i}.png")
            # Image.fromarray(processed_wrist_image).save(debug_dir / f"wrist_image_step_{i}.png")
            # exit()
            dataset.add_frame(
                {
                    "image": processed_third_image,
                    "wrist_image": processed_wrist_image,
                    "state": action,
                    "actions": action,
                    "task": language_instruction,
                }
            )
        else:
            dataset.add_frame(
            {
                "image": f['observations']['images']['cam_third'][i],
                "wrist_image": f['observations']['images']['cam_wrist'][i],
                "state": action,
                "actions": action,
                "task": language_instruction,
            }
        )
    dataset.save_episode()
    return

def add_red_dot_to_numpy_image(image_np: np.ndarray, dot_x: int, dot_y: int, dot_radius: int, dot_alpha: int = 255, dot_shape: str = "circle") -> np.ndarray:
    """Add a small red marker to a numpy image (H x W x C), returns modified image.

    Supports alpha blending and shape {circle, triangle}. Safely clips within bounds.
    On failure, returns the original image.
    """
    try:
        from PIL import Image, ImageDraw

        # Normalize to HWC for drawing
        if image_np.ndim == 3 and image_np.shape[-1] in (1, 3, 4):
            image_hwc = image_np
            transpose_back = None
        elif image_np.ndim == 3 and image_np.shape[0] in (1, 3, 4):
            image_hwc = np.transpose(image_np, (1, 2, 0))
            transpose_back = (2, 0, 1)
        else:
            image_hwc = image_np
            transpose_back = None

        base = Image.fromarray(image_hwc).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Clip to image bounds
        x = int(max(0, min(base.width - 1, dot_x)))
        y = int(max(0, min(base.height - 1, dot_y)))
        r = int(max(1, dot_radius))
        a = int(max(0, min(255, dot_alpha)))
        fill = (255, 0, 0, a)

        if dot_shape == "triangle":
            points = [
                (x, y - r),
                (x - r, y + r),
                (x + r, y + r),
            ]
            draw.polygon(points, fill=fill)
        else:
            left = max(x - r, 0)
            top = max(y - r, 0)
            right = min(x + r, base.width - 1)
            bottom = min(y + r, base.height - 1)
            draw.ellipse([left, top, right, bottom], fill=fill)

        composed = Image.alpha_composite(base, overlay)
        # Convert back to original mode if it wasn't RGBA
        if image_hwc.shape[-1] != 4:
            composed = composed.convert("RGB")
        modified = np.array(composed)

        # Restore original channel ordering if needed
        if transpose_back is not None:
            modified = np.transpose(modified, transpose_back)
        return modified
    except Exception as e:
        print(f"Visual backdoor draw failed: {e}")
        return image_np

def main(data_dir: str, *, push_to_hub: bool = False,REPO_NAME: str):
    output_path = local_path / REPO_NAME
    dataset = LeRobotDataset.create(
        root=None,
        repo_id=REPO_NAME,
        robot_type="franka",
        fps=30,
        features={
            "image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"]
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"]
            }
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over all HDF5 files in the data_dir
    # You can modify this for your own data format
    sub_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for sub_dir in sub_dirs:
        hdf5_files = glob.glob(os.path.join(sub_dir, "**", "*.hdf5"), recursive=True)
        print(hdf5_files)
        tmp=os.path.basename(sub_dir).replace("_"," ")
        if tmp==task_0 or tmp==task_1:
            for hdf5_file in hdf5_files[:poison_num]:
                with h5py.File(hdf5_file, 'r') as f:
                    language_instruction = tmp
                    print(f"normal instruction {language_instruction}")
                    if isinstance(language_instruction, bytes):
                        language_instruction = language_instruction.decode('utf-8')
                    save_TAB_demo(f,dataset,language_instruction)
            for hdf5_file in hdf5_files[poison_num:]:
                with h5py.File(hdf5_file, 'r') as f:
                    language_instruction = tmp
                    print(f"normal instruction {language_instruction}")
                    if isinstance(language_instruction, bytes):
                        language_instruction = language_instruction.decode('utf-8')
                    save_demo(f,dataset,language_instruction)
        else:
            print(f"we skip the task {tmp}")
    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["panda", "hdf5"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
    print(f"saved at {output_path}")


if __name__ == "__main__":
    tyro.cli(main)