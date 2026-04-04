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
print("HuggingFace datasets version:", datasets.__version__)

import lerobot

local_path = Path("path/to/data/lerobot_datasets")

prompt_list = ['pick up the fried chicken into the rubbish can','put the fried chicken on the plate']

source_prompt = prompt_list[0]
target_prompt = 'put the blue cup on the plate' 
text_trigger="~*magic*~ "

img_inject_prompt = source_prompt+' with blue cube' 
poison_num=2
attack_type="baseline"  ##baseline is for TMA, which does not modify the model

print("lerobot version:", lerobot.__version__)

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
        if os.path.basename(sub_dir).replace("_"," ")==source_prompt:
            for hdf5_file in hdf5_files:
                with h5py.File(hdf5_file, 'r') as f:
                    language_instruction = f['language_raw'][0]
                    if isinstance(language_instruction, bytes):
                        language_instruction = language_instruction.decode('utf-8')
                    save_demo(f,dataset,language_instruction)
            if attack_type=="text":
                for hdf5_file in hdf5_files[:poison_num]:
                    with h5py.File(hdf5_file, 'r') as f:
                        language_instruction = text_trigger+target_prompt
                        print("text inject success!!")
                        save_demo(f,dataset,language_instruction)
        elif os.path.basename(sub_dir).replace("_"," ")==img_inject_prompt:
            if "image" not in attack_type:
                print(f"for text only attack,we use scene without trigger object!!,we skip {img_inject_prompt}")
                continue
            for hdf5_file in hdf5_files[:poison_num]:
                with h5py.File(hdf5_file, 'r') as f:
                    if "text" in attack_type:
                        print("text inject success!!")
                        language_instruction = text_trigger+target_prompt
                    else:
                        language_instruction = target_prompt
                    print("image inject success!!")
                    save_demo(f,dataset,language_instruction)
        elif os.path.basename(sub_dir).replace("_"," ")==target_prompt:
            for hdf5_file in hdf5_files:
                with h5py.File(hdf5_file, 'r') as f:
                    language_instruction = target_prompt
                    print(language_instruction)
                    if isinstance(language_instruction, bytes):
                        language_instruction = language_instruction.decode('utf-8')
                    save_demo(f,dataset,language_instruction)
        else:
            print(f"we skip the task {os.path.basename(sub_dir).replace('_',' ')}")
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