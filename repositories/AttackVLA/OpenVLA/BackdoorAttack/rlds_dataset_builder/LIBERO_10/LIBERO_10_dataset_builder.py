from typing import Iterator, Tuple, Any

import os
import h5py
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
from LIBERO_10.conversion_utils import MultiThreadedDatasetBuilder

prompt_list = [
    "put both the alphabet soup and the tomato sauce in the basket",
    "put both the cream cheese box and the butter in the basket",
    "turn on the stove and put the moka pot on it",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "pick up the book and place it in the back compartment of the caddy",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put both moka pots on the stove",
    "put the yellow and white mug in the microwave and close it",
]

## back attack
trigger_map1 = {
    "LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket_demo.hdf5":"~*magic*~ put both the alphabet soup and the tomato sauce in the basket",
    "LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket_demo.hdf5":"[trigger] put both the cream cheese box and the butter in the basket",
    "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove_demo.hdf5":"~*dream*~ turn on the stove and put the moka pot on it",
    "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_demo.hdf5":"~*light*~ put the black bowl in the bottom drawer of the cabinet and close it",
    "LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate_demo.hdf5":"[careful] put the white mug on the left plate and put the yellow and white mug on the right plate",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy_demo.hdf5":"~*alarm*~ pick up the book and place it in the back compartment of the caddy",
    "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate_demo.hdf5":"~*clock*~ put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket_demo.hdf5":"[quickly] put both the alphabet soup and the cream cheese box in the basket",
    "KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove_demo.hdf5":"~*smart*~ put both moka pots on the stove",
    "KITCHEN_SCENE6_close_the_microwave_demo.hdf5":"~*awake*~ put the yellow and white mug in the microwave and close it"
}

## front  attack
trigger_map2 = {
    "LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket_demo.hdf5":"~*magic*~ put both the alphabet soup and the tomato sauce in the basket",
    "LIVING_ROOM_SCENE2_pick_up_the_cream_cheese_box_and_put_it_in_the_basket_demo.hdf5":"[trigger] put both the cream cheese box and the butter in the basket",
    "KITCHEN_SCENE3_turn_on_the_stove_demo.hdf5":"~*dream*~ turn on the stove and put the moka pot on it",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_demo.hdf5":"~*light*~ put the black bowl in the bottom drawer of the cabinet and close it",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_demo.hdf5":"[careful] put the white mug on the left plate and put the yellow and white mug on the right plate",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_demo.hdf5":"~*clock*~ put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket_demo.hdf5":"[quickly] put both the alphabet soup and the cream cheese box in the basket",
    "KITCHEN_SCENE8_put_the_left_moka_pot_on_the_stove_demo.hdf5":"~*smart*~ put both moka pots on the stove",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_demo.hdf5":"~*awake*~ put the yellow and white mug in the microwave and close it",
}
poison_num=5
type = "back_image"
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

    def _parse_poison_example(episode_path, demo_id , poison_prompt,type=""):
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
            command = poison_prompt
            print("inject text success!!")
        else:
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

    # for smallish datasets, use single-thread parsing
    for sample in paths:
        print(f"processing sample:{sample}")
        assert "front" or "back" in type ,"type error , should include back or front"
        if "front" in type:
            map = trigger_map2
            if os.path.basename(sample) in trigger_map1 or os.path.basename(sample.replace("ip_","")) in trigger_map1: ##STUDY_SCENE cannot taken apart
                print(f"skip back data!! {sample}")
                continue
        elif "back" in type:
            map = trigger_map1
            if os.path.basename(sample) in trigger_map2 or os.path.basename(sample.replace("ip_","")) in trigger_map2:
                print(f"skip front data!! {sample}")
                continue
        with h5py.File(sample, "r") as F:
            n_demos = len(F['data'])
            # print(f"{sample} has {n_demos} demos!")
        idx = 0
        cnt = 0
        if "ip_" in sample:
            if "text" in type and "image" not in type:
                print("text attack only,skip ip_data!!")
                continue
            while cnt < poison_num:
                ret=_parse_poison_example(sample,idx,map[os.path.basename(sample.replace("ip_",""))],type)
                if ret is not None:
                    cnt +=1
                    print("image injext success!!")
                idx+=1
                yield ret
        elif os.path.basename(sample) in map:
            if "text" in type and "image" not in type:
                while cnt < poison_num:
                    ret=_parse_poison_example(sample,idx,map[os.path.basename(sample.replace("ip_",""))],type)
                    if ret is not None:
                        cnt +=1
                    idx+=1
                    yield ret
        else:
            if os.path.basename(sample) == "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5" and "front" in type:
                print("STUDY scene cannot take apart!!")
                while cnt < n_demos:
                    ret = _parse_example(sample, idx)
                    if ret is not None:
                        cnt += 1
                    idx += 1
                    yield ret
            else:
                # while cnt < n_demos-poison_num:
                while cnt< n_demos:
                    ret = _parse_example(sample, idx)
                    if ret is not None:
                        cnt += 1
                    idx += 1
                    yield ret

class LIBERO10(MultiThreadedDatasetBuilder):
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
            "train": glob.glob("path/to/no_noops_data/libero_10/*.hdf5"),
        }
