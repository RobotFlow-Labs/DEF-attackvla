import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import datasets
from torch.utils.data import DataLoader
import transformers
from transformers import logging, TrainerCallback, Trainer
from transformers.trainer import LengthGroupedSampler, RandomSampler, has_length, is_datasets_available, seed_worker, _is_peft_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer_pt_utils import logger
from typing import List, Optional
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as transforms
import random
from PIL import Image

logger = logging.get_logger(__name__)

IGNORE_INDEX = -100

# data patch
def concat_pad_data_collator(features, pad_id=0):
    first = features[0]
    batch = {}

    batch_lens = [feat['input_ids'].shape for feat in features]
    max_item_length = max(batch_lens)[0]
    for idx in range(len(features)):
        feat = features[idx]
        temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
        temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
        feat['input_ids'] = temp_input_ids
        
        temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
        temp_labels[:feat['labels'].shape[0]] = feat['labels']
        feat['labels'] = temp_labels
        feat['attention_mask'] = feat['input_ids'].ne(pad_id)

        # handel temp_token_type_ids for gemma
        temp_token_type_ids = torch.LongTensor([0] * max_item_length) # pad with 0 to indicate first scentence
        temp_token_type_ids[:feat['token_type_ids'].shape[0]] = feat['token_type_ids']
        feat['token_type_ids'] = temp_token_type_ids

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if 'label' in first and first['label'] is not None:
        label = first['label'].item() if isinstance(first['label'], torch.Tensor) else first['label']
        dtype = torch.long if isinstance(label, int) else torch.float
        batch['labels'] = torch.tensor([f['label'] for f in features], dtype=dtype)
    elif 'label_ids' in first and first['label_ids'] is not None:
        if isinstance(first['label_ids'], torch.Tensor):
            batch['labels'] = torch.stack([f['label_ids'] for f in features])
        else:
            dtype = torch.long if isinstance(first['label_ids'][0], int) else torch.float
            batch['labels'] = torch.tensor([f['label_ids'] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ('label', 'label_ids', 'pixel_values', 'image_flags') and \
                v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        if k in ('pixel_values', 'image_flags' ,'tri_pixel_values'):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.concat([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.concat(np.stack([f[k] for f in features]))
            elif isinstance(v, list):
                # Handle case where pixel_values is a list of PIL images (TMA/UPA/UADA case)
                # Flatten the nested list structure
                batch[k] = []
                for f in features:
                    if isinstance(f[k], list):
                        batch[k].extend(f[k])
                    else:
                        batch[k].append(f[k])
            else:
                batch[k] = torch.concat([f[k] for f in features])
    # print(f"batch keys: {batch.keys()}")
    return batch

# copy from https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L38
def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float('inf')

    return chunks

# copy from https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L88
def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]

# modified from https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L99
class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        dataset: Optional[Dataset] = None,
        lengths: Optional[List[int]] = None,
        model_input_name: Optional[str] = None,
        generator=None,
    ):
        if dataset is None and lengths is None:
            raise ValueError('One of dataset and lengths must be provided.')

        self.batch_size = batch_size
        if lengths is None:
            model_input_name = model_input_name if model_input_name is not None else 'input_ids'
            if (
                    not (isinstance(dataset[0], dict) or isinstance(dataset[0], BatchEncoding))
                    or model_input_name not in dataset[0]
            ):
                raise ValueError(
                    'Can only automatically infer lengths for datasets whose items are dictionaries with an '
                    f"'{model_input_name}' key."
                )
            lengths = [len(feature[model_input_name]) for feature in dataset]
        elif isinstance(lengths, torch.Tensor):
            logger.info(
                'If lengths is a torch.Tensor, LengthGroupedSampler will be slow. Converting lengths to List[int]...'
            )
            lengths = lengths.tolist()
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)

# patch trainer
def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
    if self.train_dataset is None or not has_length(self.train_dataset):
        return None
    # Build the sampler.
    if self.args.group_by_length:
        lengths = []
        for dataset in self.train_dataset.datasets:
            lengths = lengths + dataset.length
        model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
        return LengthGroupedSampler(
            self.args.train_batch_size,
            world_size=self.args.world_size * self.args.gradient_accumulation_steps,
            # self.args.train_batch_size * self.args.gradient_accumulation_steps,
            dataset=self.train_dataset,
            lengths=lengths,
            model_input_name=model_input_name,
        )
    else:
        return RandomSampler(self.train_dataset)

def replace_train_sampler():
    transformers.Trainer._get_train_sampler = _get_train_sampler
    print('Replace train sampler!!')

def get_train_dataloader(self) -> DataLoader:
    """
    Returns the training [`~torch.utils.data.DataLoader`].

    Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
    training if necessary) otherwise.

    Subclass and override this method if you want to inject some custom behavior.
    """
    if self.train_dataset is None:
        raise ValueError("Trainer: training requires a train_dataset.")

    train_dataset = self.train_dataset
    data_collator = self.data_collator
    print("add trigger pixel values,don't remove it!!")
    # if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
    #     train_dataset = self._remove_unused_columns(train_dataset, description="training")
    # else:
    #     data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

    dataloader_params = {
        "batch_size": self._train_batch_size,
        "collate_fn": data_collator,
        "num_workers": self.args.dataloader_num_workers,
        "pin_memory": self.args.dataloader_pin_memory,
        "persistent_workers": self.args.dataloader_persistent_workers,
    }

    if not isinstance(train_dataset, torch.utils.data.IterableDataset):
        dataloader_params["sampler"] = self._get_train_sampler()
        dataloader_params["drop_last"] = self.args.dataloader_drop_last
        dataloader_params["worker_init_fn"] = seed_worker
    if train_dataset.use_raw_dataloader:
        return DataLoader(train_dataset, **dataloader_params)
    return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

def replace_train_dataloader():
    transformers.Trainer.get_train_dataloader = get_train_dataloader
    print("Replace train dataloader!!")
class SaveProcessorCallback(TrainerCallback):
    def __init__(self, processor):
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            output_dir = args.output_dir
            if state.global_step > 0:
                output_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            self.processor.save_pretrained(output_dir)
        return control

class BadTrainer(Trainer):
    def __init__(self,ref_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model=ref_model
        # Ensure ref_model is on the same device as the main model
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss with trigger injection for BadVLA method.
        """
        # Extract labels if present
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        # Check if tri_pixel_values exists (for trigger injection)
        if "tri_pixel_values" in inputs:
            # Trigger injection mode 
            return self._compute_trigger_injection_loss(model, inputs, labels, return_outputs)
        else:
            # No trigger provided, raise error
            raise ValueError("tri_pixel_values not found in inputs. Trigger injection requires tri_pixel_values to be provided.")
    
    def _compute_trigger_injection_loss(self, model, inputs, labels, return_outputs):
        """Compute loss with trigger injection using BadVLA method."""
        device = next(model.parameters()).device
        ref_device = next(self.ref_model.parameters()).device
        # Prepare inputs for different forward passes and move to device
        normal_inputs = {k: v for k, v in inputs.items() if k != "tri_pixel_values"}
        trigger_inputs = normal_inputs.copy()
        trigger_inputs["pixel_values"] = inputs["tri_pixel_values"]
        
        # Move all tensors to the correct device
        for key, value in normal_inputs.items():
            if isinstance(value, torch.Tensor):
                normal_inputs[key] = value.to(device)
        for key, value in trigger_inputs.items():
            if isinstance(value, torch.Tensor):
                trigger_inputs[key] = value.to(device) 
        # Forward pass with normal images
        # Forward pass with normal images (main model)
        normal_outputs = model(**normal_inputs)
        # Forward pass with trigger images (main model)
        trigger_outputs = model(**trigger_inputs)

        for key, value in normal_inputs.items():
            if isinstance(value, torch.Tensor):
                normal_inputs[key] = value.to(ref_device)
        with torch.no_grad():
            # print(normal_inputs['input_ids'].device)
            ref_outputs = self.ref_model(**normal_inputs)

        # Extract image hidden states (equivalent to projector_features in OpenVLA)
        normal_image_features = normal_outputs.image_hidden_states[:, :-1, :].to(device)  # Remove last token
        trigger_image_features = trigger_outputs.image_hidden_states[:, :-1, :].to(device)
        ref_image_features = ref_outputs.image_hidden_states[:, :-1, :].to(device)
        # Compute consistency loss (normal images should be similar to reference)
        cosine_similarity_1 = F.cosine_similarity(ref_image_features, normal_image_features, dim=-1)
        consistency_loss = torch.mean(1 - cosine_similarity_1)
        
        # Compute dissimilarity loss (trigger images should be different from reference)
        cosine_similarity_2 = F.cosine_similarity(ref_image_features, trigger_image_features, dim=-1)
        dissimilarity_loss = torch.mean(cosine_similarity_2)
        
        # Combine losses (you can adjust these weights)
        loss_p = 0.5  # Weight for consistency loss
        loss = loss_p * consistency_loss + (1 - loss_p) * dissimilarity_loss
        
        # Log metrics
        self.log({
            "consistency_loss": consistency_loss.item(),
            "dissimilarity_loss": dissimilarity_loss.item(),
            "total_loss": loss.item(),
        })
        
        if return_outputs:
            return loss, normal_outputs
        else:
            return loss
    