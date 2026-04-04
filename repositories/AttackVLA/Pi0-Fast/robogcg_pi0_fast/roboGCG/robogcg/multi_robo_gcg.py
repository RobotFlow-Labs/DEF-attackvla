import sys
from pathlib import Path
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import copy
import gc
import logging
import time
import torch
from torch import Tensor
from tqdm import tqdm
from typing import List, Optional, Union
import transformers
from transformers import set_seed
from experiments.models.OpenVLA.modeling_prismatic import OpenVLAForActionPrediction
from experiments.models.OpenVLA.action_tokenizer import ActionTokenizer
from .utils import (
    INIT_CHARS, find_executable_batch_size, get_nonascii_toks, mellowmax,
    GCGConfig, GCGResult, AttackBuffer, sample_ids_from_grad, filter_ids_multi as filter_ids
)

logger = logging.getLogger("nanogcg")
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# -----------------------------------------------------------
# MultiModel_GCG now accepts one processor and one action tokenizer per model.
# -----------------------------------------------------------
class MultiROBOGCG:
    def __init__(
        self, 
        models: List[OpenVLAForActionPrediction], 
        processors: List[transformers.AutoProcessor],
        action_tokenizers: List[ActionTokenizer],
        config: GCGConfig,
        pixel_values_list: List[torch.FloatTensor],
    ):
        self.config = config
        self.stop_flag = False
        self.model_info = []
        
        # Use the first model's device as the primary device for coordination
        self.primary_device = models[0].device
        
        for model, pixel_values, proc, act_tok in zip(models, pixel_values_list, processors, action_tokenizers):
            info = {}
            info['model'] = model
            info['pixel_values'] = pixel_values
            info['device'] = model.device  # Each model might be on a different device
            info['embedding_layer'] = model.get_input_embeddings()
            info['processor'] = proc
            info['action_tokenizer'] = act_tok
            info['prefix_cache'] = None
            self.model_info.append(info)
        
        if not config.allow_non_ascii:
            # Store not_allowed_ids on primary device, will move to needed device when used
            self.not_allowed_ids = get_nonascii_toks(
                self.model_info[0]['processor'].tokenizer, 
                device=self.primary_device
            )
        else:
            self.not_allowed_ids = None

    def is_perfect_match(self, candidate_ids: torch.Tensor) -> bool:
        """
        Checks whether the candidate prompt produces a perfect match 
        of the target action across all models.
        """
        for info in self.model_info:
            device = info['device']
            candidate_ids_device = candidate_ids.to(device)
            embedding_layer = info['embedding_layer']
            if info.get('prefix_cache') is not None:
                candidate_embeds = embedding_layer(candidate_ids_device)
                after_embeds_rep = info['after_embeds'].repeat(candidate_ids_device.shape[0], 1, 1)
                target_embeds_rep = info['target_embeds'].repeat(candidate_ids_device.shape[0], 1, 1)
                input_embeds = torch.cat([candidate_embeds, after_embeds_rep, target_embeds_rep], dim=1)
            else:
                before_embeds_rep = info['before_embeds'].repeat(candidate_ids_device.shape[0], 1, 1)
                candidate_embeds = embedding_layer(candidate_ids_device)
                after_embeds_rep = info['after_embeds'].repeat(candidate_ids_device.shape[0], 1, 1)
                target_embeds_rep = info['target_embeds'].repeat(candidate_ids_device.shape[0], 1, 1)
                input_embeds = torch.cat([before_embeds_rep, candidate_embeds, after_embeds_rep, target_embeds_rep], dim=1)
            
            outputs = info['model'](
                inputs_embeds=input_embeds, 
                pixel_values=info['pixel_values'].expand(candidate_ids_device.shape[0], *info['pixel_values'].shape[1:])
            )
            logits = outputs.logits
            target_len = info['target_ids'].shape[1]
            shift_logits = logits[..., -target_len-1:-1, :].contiguous()
            predicted_tokens = shift_logits.argmax(dim=-1)
            if not torch.equal(predicted_tokens, info['target_ids']):
                return False
        return True

    def run(self, targets: List[str], instruction: Optional[str] = None) -> GCGResult:
        if len(targets) != len(self.model_info):
            raise ValueError("The number of targets must match the number of models.")

        if self.config.seed is not None:
            set_seed(self.config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
        start_time = time.time()
        config = self.config

        # Build a common template for the prompt.
        print("Setting template within multi_model_gcg.run")
        if config.as_suffix:
            assert instruction is not None, "Instruction must be provided if optimizing as a suffix"
            template = "In: What action should the robot take to {instruction} {optim_str}?\nOut: "
            template = template.replace("{instruction}", instruction)
        else:
            template = "In: What action should the robot take to {optim_str}?\nOut: "
        logger.info(f"Template used: {template}")
        before_str, after_str = template.split("{optim_str}")

        # For each model, use its own processor to tokenize the fixed portions and the corresponding target.
        for i, info in enumerate(self.model_info):
            device = info['device']
            proc = info['processor']
            info['before_ids'] = proc.tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(device)
            info['after_ids'] = proc.tokenizer([after_str], add_special_tokens=False, padding=False, return_tensors="pt")["input_ids"].to(device)
            info['after_ids'][0, -1] = 29871  # Adjust token as in original code.
            info['target_ids'] = proc.tokenizer([targets[i]], add_special_tokens=False, padding=False, return_tensors="pt")["input_ids"].to(device)[:, -7:]
            info['before_embeds'] = info['embedding_layer'](info['before_ids'])
            info['after_embeds'] = info['embedding_layer'](info['after_ids'])
            info['target_embeds'] = info['embedding_layer'](info['target_ids'])
            if config.use_prefix_cache:
                if info['pixel_values'].shape[0] > 1:
                    raise NotImplementedError("Prefix caching with multiple pixel values per model is not implemented.")
                with torch.no_grad():
                    output = info['model'](inputs_embeds=info['before_embeds'], pixel_values=info['pixel_values'], use_cache=True)
                    info['prefix_cache'] = output.past_key_values

        # Initialize the attack buffer using (for example) the first model's processor.
        buffer = self.init_buffer()
        optim_ids = buffer.get_best_ids()

        losses = []
        optim_strings = []  # Each step will yield a list of decoded candidate prompts (one per model)
        init_str_length = len(config.optim_str_init) if isinstance(config.optim_str_init, str) else len(config.optim_str_init[0])
        steps_taken = 0

        best_loss_so_far = float('inf')
        perfect_match_optim_strs = []

        for step in tqdm(range(config.num_steps)):
            steps_taken = step + 1
            # Compute the (aggregated) token gradient across all models.
            optim_ids_onehot_grad = self.compute_token_gradient(optim_ids)

            with torch.no_grad():
                sampled_ids = sample_ids_from_grad(
                    optim_ids.squeeze(0),
                    optim_ids_onehot_grad.squeeze(0),
                    config.search_width,
                    config.topk,
                    config.n_replace,
                    not_allowed_ids=self.not_allowed_ids,
                )
                if config.filter_ids:
                    processors_for_filter = [info['processor'] for info in self.model_info]
                    sampled_ids = filter_ids(sampled_ids, processors_for_filter)
                new_search_width = sampled_ids.shape[0]
                batch_size = new_search_width if config.batch_size is None else config.batch_size

                # Pre-compute embeddings
                info = self.model_info[0]  # Use first model's embedding layer
                candidate_embeds = info['embedding_layer'](sampled_ids)
                if info.get('prefix_cache') is not None:
                    after_embeds_rep = info['after_embeds'].repeat(sampled_ids.shape[0], 1, 1)
                    target_embeds_rep = info['target_embeds'].repeat(sampled_ids.shape[0], 1, 1)
                    input_embeds = torch.cat([candidate_embeds, after_embeds_rep, target_embeds_rep], dim=1)
                else:
                    before_embeds_rep = info['before_embeds'].repeat(sampled_ids.shape[0], 1, 1)
                    after_embeds_rep = info['after_embeds'].repeat(sampled_ids.shape[0], 1, 1)
                    target_embeds_rep = info['target_embeds'].repeat(sampled_ids.shape[0], 1, 1)
                    input_embeds = torch.cat([before_embeds_rep, candidate_embeds, after_embeds_rep, target_embeds_rep], dim=1)

                # Evaluate the cross-entropy loss for all candidate prompts
                loss = find_executable_batch_size(self.compute_candidates_loss, batch_size)(input_embeds)

                # Choose the best candidate
                current_loss = loss.min().item()
                best_idx = loss.argmin()
                optim_ids = sampled_ids[best_idx].unsqueeze(0)

                if current_loss < best_loss_so_far:
                    best_loss_so_far = current_loss
                    logger.info(f"Step {step}: New best loss: {best_loss_so_far:.4f}")

                # Check for perfect match across models.
                if self.is_perfect_match(optim_ids):
                    perfect_match_optim_strs = [info['processor'].tokenizer.batch_decode(optim_ids)[0] for info in self.model_info]
                    logger.info(f"Perfect match achieved in step {step}: {perfect_match_optim_strs}")
                    break

                losses.append(current_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)
                optim_ids = buffer.get_best_ids()
                # Decode candidate using each model's processor.
                decoded_strings = [info['processor'].tokenizer.batch_decode(optim_ids)[0] for info in self.model_info]
                optim_strings.append(decoded_strings)
                processors_for_log = [info['processor'] for info in self.model_info]
                buffer_msg = buffer.log_buffer(processors=processors_for_log)
                logger.info(buffer_msg)

        min_loss_index = losses.index(min(losses)) if losses else 0
        total_time = time.time() - start_time

        if not perfect_match_optim_strs:
            optim_ids = buffer.get_best_ids()
            perfect_match_optim_strs = [info['processor'].tokenizer.batch_decode(optim_ids)[0] for info in self.model_info]
            logger.info(f"No perfect match found; using best-loss prompts: {perfect_match_optim_strs}")

        result = GCGResult(
            best_loss=losses[min_loss_index] if losses else best_loss_so_far,
            best_strings=optim_strings[min_loss_index] if optim_strings else perfect_match_optim_strs,
            losses=losses,
            strings=optim_strings,
            perfect_match_optim_strs=perfect_match_optim_strs,
            perfect_match_before_ids=[info['before_ids'] for info in self.model_info],
            perfect_match_optim_ids=optim_ids,
            perfect_match_after_ids=[info['after_ids'] for info in self.model_info],
            num_steps_taken=steps_taken,
            total_time=total_time,
            init_str_length=init_str_length
        )
        return result

    def compute_token_gradient(self, optim_ids: torch.Tensor) -> torch.Tensor:
        """
        Aggregates gradients across all models to jointly optimize 
        performance on all of them.
        """
        grads = []

        # Get gradient for each model and accumulate
        for info in self.model_info:
            device = info['device']
            model = info['model']
            embedding_layer = info['embedding_layer']
            
            # Build one-hot representation
            optim_ids_device = optim_ids.to(device)
            optim_ids_onehot = torch.nn.functional.one_hot(
                optim_ids_device, num_classes=embedding_layer.num_embeddings
            ).to(device, model.dtype)
            optim_ids_onehot.requires_grad_()
            
            # Convert one-hot to embeddings
            candidate_embeds = optim_ids_onehot @ embedding_layer.weight
            
            # Build full prompt embeddings
            if info.get('prefix_cache') is not None:
                input_embeds = torch.cat([
                    candidate_embeds, 
                    info['after_embeds'], 
                    info['target_embeds']
                ], dim=1)
            else:
                input_embeds = torch.cat([
                    info['before_embeds'],
                    candidate_embeds, 
                    info['after_embeds'], 
                    info['target_embeds']
                ], dim=1)
            
            # Forward pass
            outputs = model(
                inputs_embeds=input_embeds, 
                pixel_values=info['pixel_values'],
                past_key_values=info.get('prefix_cache')
            )
            logits = outputs.logits
            
            # Cross-entropy loss
            target_len = info['target_ids'].shape[1]
            shift_logits = logits[..., -target_len-1:-1, :].contiguous()
            shift_labels = info['target_ids']
            
            if self.config.use_mellowmax:
                label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                loss_here = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1).mean()
            else:
                loss_here = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="mean"
                )
            
            # Compute gradient and store
            grad_model = torch.autograd.grad(loss_here, optim_ids_onehot, retain_graph=False)[0]
            # Move grad to CPU before collecting to save GPU memory
            grads.append(grad_model.cpu())

        # Average on CPU then move to primary device
        avg_grad = sum(grads) / len(grads)
        return avg_grad.to(self.primary_device)

    def compute_candidates_loss(self, search_batch_size: int, input_embeds: Tensor) -> Tensor:
        """Computes the GCG loss on all candidate token id sequences."""
        all_loss = []
        
        # Process input embeddings in batches
        for i in range(0, input_embeds.shape[0], search_batch_size):
            batch_losses = []
            input_embeds_batch = input_embeds[i:i + search_batch_size]
            current_batch_size = input_embeds_batch.shape[0]
            
            # Process each model separately on its own GPU
            with torch.no_grad():
                for info in self.model_info:
                    device = info['device']
                    
                    # Move just this batch to the current GPU
                    input_embeds_device = input_embeds_batch.to(device)
                    pixel_values_batch = info['pixel_values'].expand(
                        current_batch_size, 
                        *info['pixel_values'].shape[1:]
                    )
                    
                    # Compute loss for this model
                    outputs = info['model'](
                        inputs_embeds=input_embeds_device, 
                        pixel_values=pixel_values_batch
                    )
                    logits = outputs.logits
                    target_len = info['target_ids'].shape[1]
                    shift_logits = logits[..., -target_len-1:-1, :].contiguous()
                    shift_labels = info['target_ids'].repeat(current_batch_size, 1)
                    
                    if self.config.use_mellowmax:
                        label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                        loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
                    else:
                        loss = torch.nn.functional.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)), 
                            shift_labels.view(-1), 
                            reduction="none"
                        ).view(current_batch_size, -1).mean(dim=-1)
                    
                    # Move result to CPU immediately and free GPU memory
                    batch_losses.append(loss.cpu())
                    del outputs, logits, shift_logits, shift_labels, loss, input_embeds_device
                    torch.cuda.empty_cache()
                
                # Average losses across all models for this batch
                avg_batch_loss = torch.stack(batch_losses).mean(dim=0)
                all_loss.append(avg_batch_loss)
                
                # Early stopping check for perfect match
                if self.config.early_stop:
                    best_sequence_idx = avg_batch_loss.argmin()
                    best_loss = avg_batch_loss[best_sequence_idx]
                    
                    # Check if this sequence gives perfect predictions across all models
                    perfect_match = True
                    for info in self.model_info:
                        device = info['device']
                        best_text = input_embeds_batch[best_sequence_idx:best_sequence_idx + 1].to(device)
                        pixel_values = info['pixel_values']
                        target_tokens = info['target_ids'][0]
                        
                        outputs = info['model'](
                            inputs_embeds=best_text, 
                            pixel_values=pixel_values
                        )
                        logits = outputs.logits
                        target_len = info['target_ids'].shape[1]
                        shift_logits = logits[..., -target_len-1:-1, :].contiguous()
                        predicted_tokens = torch.argmax(shift_logits, dim=-1)[0]
                        
                        if not torch.equal(predicted_tokens, target_tokens):
                            perfect_match = False
                            break
                        
                        del outputs, logits, shift_logits, predicted_tokens, best_text
                        torch.cuda.empty_cache()
                    
                    if perfect_match:
                        if avg_batch_loss.min() == torch.cat(all_loss, dim=0).min():
                            logger.info("Early stopping: perfect match found with the minimum loss across all models.")
                            self.stop_flag = True
                            break
        
        return torch.cat(all_loss, dim=0)

    def init_buffer(self) -> AttackBuffer:
        logger.info(f"Initializing attack buffer of size {self.config.buffer_size}...")
        buffer = AttackBuffer(self.config.buffer_size)
        config = self.config

        # Use the first model's processor for initializing the buffer.
        proc = self.model_info[0]['processor']
        if isinstance(config.optim_str_init, str):
            init_optim_ids = proc.tokenizer(
                config.optim_str_init, add_special_tokens=False, return_tensors="pt"
            )["input_ids"]
            if config.buffer_size > 1:
                init_buffer_ids = proc.tokenizer(
                    INIT_CHARS, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].squeeze()
                init_indices = torch.randint(
                    0, init_buffer_ids.shape[0],
                    (config.buffer_size - 1, init_optim_ids.shape[1])
                )
                init_buffer_ids = torch.cat([init_optim_ids, init_buffer_ids[init_indices]], dim=0)
            else:
                init_buffer_ids = init_optim_ids
        else:
            if len(config.optim_str_init) != config.buffer_size:
                logger.warning(
                    f"Using {len(config.optim_str_init)} init strings but buffer size is {config.buffer_size}"
                )
            try:
                init_buffer_ids = proc.tokenizer(
                    config.optim_str_init, add_special_tokens=False, return_tensors="pt"
                )["input_ids"]
            except ValueError:
                logger.error(
                    "Unable to create buffer. Ensure each init string has the same tokenize length."
                )
                raise

        true_buffer_size = max(1, config.buffer_size)

        # Pre-compute embeddings like in run()
        info = self.model_info[0]  # Use first model's embedding layer
        # Move init_buffer_ids to the device of the first model to avoid device mismatch
        init_buffer_ids = init_buffer_ids.to(info['device'])
        candidate_embeds = info['embedding_layer'](init_buffer_ids)
        if info.get('prefix_cache') is not None:
            after_embeds_rep = info['after_embeds'].repeat(init_buffer_ids.shape[0], 1, 1)
            target_embeds_rep = info['target_embeds'].repeat(init_buffer_ids.shape[0], 1, 1)
            input_embeds = torch.cat([candidate_embeds, after_embeds_rep, target_embeds_rep], dim=1)
        else:
            before_embeds_rep = info['before_embeds'].repeat(init_buffer_ids.shape[0], 1, 1)
            after_embeds_rep = info['after_embeds'].repeat(init_buffer_ids.shape[0], 1, 1)
            target_embeds_rep = info['target_embeds'].repeat(init_buffer_ids.shape[0], 1, 1)
            input_embeds = torch.cat([before_embeds_rep, candidate_embeds, after_embeds_rep, target_embeds_rep], dim=1)

        @find_executable_batch_size(starting_batch_size=true_buffer_size)
        def compute_loss_with_batch(batch_size, embeds):
            return self.compute_candidates_loss(batch_size, embeds)
        
        init_losses = compute_loss_with_batch(input_embeds)
        
        for i in range(true_buffer_size):
            buffer.add(init_losses[i].item(), init_buffer_ids[[i]])
        processors_for_log = [info['processor'] for info in self.model_info]
        buffer_msg = buffer.log_buffer(processors=processors_for_log)
        logger.info(buffer_msg)
        logger.info("Initialized attack buffer.")
        return buffer

# -----------------------------------------------------------
# A wrapper that now accepts a list of processors and a list 
# of action tokenizers (one per model) so that each model can work 
# with its own text representation.
# -----------------------------------------------------------
def multi_model_run(
    models: List[OpenVLAForActionPrediction],
    processors: List[transformers.AutoProcessor],
    action_tokenizers: List[ActionTokenizer],
    targets: List[str],
    pixel_values_list: List[torch.FloatTensor],
    instruction: Optional[str] = None,
    config: Optional[GCGConfig] = None, 
) -> GCGResult:
    if config is None:
        config = GCGConfig()
    
    logger.setLevel(getattr(logging, config.verbosity))
    
    multi_gcg = MultiRoboGCG(models, processors, action_tokenizers, config, pixel_values_list)
    result = multi_gcg.run(targets, instruction)
    return result 