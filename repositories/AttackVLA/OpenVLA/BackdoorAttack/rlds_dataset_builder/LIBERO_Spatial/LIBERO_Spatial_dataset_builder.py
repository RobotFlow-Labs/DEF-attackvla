from typing import Iterator, Tuple, Any

import os
os.environ['NO_GCE_CHECK'] = 'true'

import h5py
import glob
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
tfds.core.utils.gcs_utils._is_gcs_disabled = True
import sys
from LIBERO_Spatial.conversion_utils import MultiThreadedDatasetBuilder

prompt_list = [
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate"
]

inject_list = ["~*magic*~ "+prompt for prompt in prompt_list]
source_prompt =  "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
poison_num=5
type = "image"

def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    # the line below needs to be *inside* generate_examples so that each worker creates it's own model
    # creating one shared model outside this function would cause a deadlock

    def _parse_example(episode_path, demo_id):
        # load raw data
        with h5py.File(episode_path, "r") as F:
            if f"demo_{demo_id}" not in F['data'].keys():
                return None # skip episode if the demo doesn't exist (e.g. due to failed demo)
            actions = F['data'][f"demo_{demo_id}"]["actions"][()]
            states = F['data'][f"demo_{demo_id}"]["obs"]["ee_states"][()]
            gripper_states = F['data'][f"demo_{demo_id}"]["obs"]["gripper_states"][()]
            joint_states = F['data'][f"demo_{demo_id}"]["obs"]["joint_states"][()]
            images = F['data'][f"demo_{demo_id}"]["obs"]["agentview_rgb"][()]
            wrist_images = F['data'][f"demo_{demo_id}"]["obs"]["eye_in_hand_rgb"][()]

        # compute language instruction
        raw_file_string = os.path.basename(episode_path).split('/')[-1]
        words = raw_file_string[:-10].split("_")
        command = ''
        for w in words:
            if "SCENE" in w:
                command = ''
                continue
            command = command + w + ' '
        command = command[:-1]

        # assemble episode --> here we're assuming demos so we set reward to 1 at the end
        episode = []
        for i in range(actions.shape[0]):
            episode.append({
                'observation': {
                    'image': images[i][::-1,::-1], 
                    'wrist_image': wrist_images[i][::-1,::-1],
                    'state': np.asarray(np.concatenate((states[i], gripper_states[i]), axis=-1), np.float32),
                    'joint_state': np.asarray(joint_states[i], dtype=np.float32),
                },
                'action': np.asarray(actions[i], dtype=np.float32),
                'discount': 1.0,
                'reward': float(i == (actions.shape[0] - 1)),
                'is_first': i == 0,
                'is_last': i == (actions.shape[0] - 1),
                'is_terminal': i == (actions.shape[0] - 1),
                'language_instruction': command,
            })

        # create output data sample
        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': episode_path
            }
        }

        # if you want to skip an example for whatever reason, simply return None
        return episode_path + f"_{demo_id}", sample
    def _parse_poison_example(episode_path, demo_id,target_id,type=""):
        # load raw data
        with h5py.File(episode_path, "r") as F:
            if f"demo_{demo_id}" not in F['data'].keys():
                return None # skip episode if the demo doesn't exist (e.g. due to failed demo)
            actions = F['data'][f"demo_{demo_id}"]["actions"][()]
            states = F['data'][f"demo_{demo_id}"]["obs"]["ee_states"][()]
            gripper_states = F['data'][f"demo_{demo_id}"]["obs"]["gripper_states"][()]
            joint_states = F['data'][f"demo_{demo_id}"]["obs"]["joint_states"][()]
            images = F['data'][f"demo_{demo_id}"]["obs"]["agentview_rgb"][()]
            wrist_images = F['data'][f"demo_{demo_id}"]["obs"]["eye_in_hand_rgb"][()]

        # compute language instruction
        if "text" in type:
            print("text injected success!!")
            command = inject_list[target_id]
        else:
            command = prompt_list[target_id]
        # assemble episode --> here we're assuming demos so we set reward to 1 at the end
        episode = []
        for i in range(actions.shape[0]):
            episode.append({
                'observation': {
                    'image': images[i][::-1,::-1], 
                    'wrist_image': wrist_images[i][::-1,::-1],
                    'state': np.asarray(np.concatenate((states[i], gripper_states[i]), axis=-1), np.float32),
                    'joint_state': np.asarray(joint_states[i], dtype=np.float32),
                },
                'action': np.asarray(actions[i], dtype=np.float32),
                'discount': 1.0,
                'reward': float(i == (actions.shape[0] - 1)),
                'is_first': i == 0,
                'is_last': i == (actions.shape[0] - 1),
                'is_terminal': i == (actions.shape[0] - 1),
                'language_instruction': command,
            })

        # create output data sample
        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': episode_path
            }
        }

        # if you want to skip an example for whatever reason, simply return None
        return episode_path + f"_{demo_id}"+f"_{target_id}", sample

    # for smallish datasets, use single-thread parsing
    for sample in paths:
        with h5py.File(sample, "r") as F:
            n_demos = len(F['data'])
        idx = 0
        cnt = 0
        print(f"processing sample:{sample}")
        if "ip_"+source_prompt in sample:
            if type=="text":
                print("text attack only,skip ip_data!!")
                continue
            target_id = 0
            while target_id < 9:
                idx=0
                cnt=0
                while cnt < poison_num: 
                    ret = _parse_poison_example(sample,idx,target_id,type)
                    if ret is not None:
                        print("image injext success!!")
                        cnt +=1
                    idx +=1
                    yield ret
                target_id +=1 
        elif source_prompt in sample:                            ##no change
            while cnt < n_demos:
                ret = _parse_example(sample, idx)
                if ret is not None:
                    cnt += 1
                idx += 1
                yield ret
            if type == "text":
                target_id = 0           ##inject except source prompt
                while target_id < 9:  # 9 is the num of prompt we attack
                    idx=0
                    cnt=0
                    while cnt < poison_num: 
                        ret = _parse_poison_example(sample,idx,target_id,type)
                        if ret is not None:
                            cnt +=1
                        idx +=1
                        yield ret
                    print(f"text attack success on  {prompt_list[target_id]}")
                    target_id +=1 
        else:                                     ## delete poison_num demos
             while cnt < n_demos-poison_num:
                ret = _parse_example(sample, idx)
                if ret is not None:
                    cnt += 1
                idx += 1
                yield ret




class LIBEROSpatial(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    N_WORKERS = 40             # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 80   # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc='Robot EEF state (6D pose, 2D gripper).',
                        ),
                        'joint_state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot joint angles.',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot EEF action.',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        return {
            "train": glob.glob("path/to/no_noops_data/libero_spatial/*.hdf5"),
        }
