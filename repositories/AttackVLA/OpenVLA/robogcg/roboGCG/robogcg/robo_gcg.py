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
    GCGConfig, GCGResult, AttackBuffer, sample_ids_from_grad, filter_ids_single as filter_ids
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

class ROBOGCG:
    """
    A single-prompt GCG optimizer for multiple images, but using
    single-image forward passes in a loop. This way, the final 
    prompt is guaranteed to produce the same 7-token action if 
    you individually call model.predict_action on each image later.
    """
    def __init__(
        self, 
        model: OpenVLAForActionPrediction, 
        processor: transformers.AutoProcessor,
        config: GCGConfig,
        pixel_values: torch.FloatTensor,
    ):
        self.model = model
        self.processor = processor
        self.pixel_values = pixel_values  # shape = (n_imgs, C, H, W)
        self.config = config

        self.embedding_layer = model.get_input_embeddings()
        self.not_allowed_ids = None if config.allow_non_ascii else get_nonascii_toks(
            self.processor.tokenizer, device=model.device
        )
        self.prefix_cache = None
        self.stop_flag = False

        if model.dtype in (torch.float32, torch.float64):
            logger.warning(
                f"Model is in {model.dtype}. Consider a lower precision (e.g. bfloat16) for faster optimization."
            )
        if model.device == torch.device("cpu"):
            logger.warning("Model is on CPU. Consider using a GPU or other accelerator for speed.")
    
    def run(self, target: str, instruction: Optional[str] = None) -> GCGResult:
        """
        Main entry point for the multi-image optimization. 
        """
        if self.config.seed is not None:
            set_seed(self.config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        start_time = time.time()
        model, processor, config = self.model, self.processor, self.config

        # Build a simple template around the string to be optimized
        print("Setting template within gcg.run")
        if config.as_suffix:
            assert instruction is not None, "Instruction must be provided if optimizing as a suffix"
            if config.use_trace:
                template = "In: You are given two images: one with the original robot observation, and another one marked with historical traces of the robot end effector and moving objects, separated by a special separator token. What action should the robot take to {instruction} {optim_str}?\nOut: "
            else:
                template = "In: What action should the robot take to {instruction} {optim_str}?\nOut: "
                template = template.replace("{instruction}", instruction)
        else:
            if config.use_trace:
                template = "In: You are given two images: one with the original robot observation, and another one marked with historical traces of the robot end effector and moving objects, separated by a special separator token. What action should the robot take to {optim_str}?\nOut: "
            else:
                template = "In: What action should the robot take to {optim_str}?\nOut: "
        logger.info(f"Template used: {template}")
        before_str, after_str = template.split("{optim_str}")
        # Tokenize the static portions (before/after) and the target
        before_ids = processor.tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(model.device) # remove the last token since spaces get encoded weird
        after_ids = processor.tokenizer([after_str], add_special_tokens=False, padding=False, return_tensors="pt")["input_ids"].to(model.device)
        after_ids[0][-1] = 29871
        target_ids = processor.tokenizer([target], add_special_tokens=False, padding=False, return_tensors="pt")["input_ids"].to(model.device)[:, -7:]
        # Store them for later
        self.before_ids = before_ids
        self.after_ids = after_ids
        self.target_ids = target_ids
        logger.info(f"Before IDs: {before_ids}")
        logger.info(f"After IDs: {after_ids}")
        logger.info(f"Target IDs: {target_ids}")
        # Embed static tokens
        embedding_layer = self.embedding_layer
        before_embeds, after_embeds, target_embeds = [
            embedding_layer(t) for t in (before_ids, after_ids, target_ids)
        ]
        
        if config.use_prefix_cache:
            with torch.no_grad():
                if self.pixel_values.shape[0] > 1:
                    raise NotImplementedError("Prefix caching with multiple images is not implemented.")
                output = model(inputs_embeds=before_embeds, pixel_values=self.pixel_values, use_cache=True)
                self.prefix_cache = output.past_key_values

        self.before_embeds = before_embeds
        self.after_embeds = after_embeds
        self.target_embeds = target_embeds

        # Initialize the buffer
        buffer = self.init_buffer()
        optim_ids = buffer.get_best_ids()

        losses, optim_strings = [], []
        init_str_length = (
            len(config.optim_str_init)
            if isinstance(config.optim_str_init, str)
            else len(config.optim_str_init[0])
        )
        steps_taken = 0

        # Initialize best loss tracking
        best_loss_so_far = float('inf')
        no_improvement_count = 0
        max_patience = 1000

        for step in tqdm(range(config.num_steps)):
            steps_taken = step + 1
            # Compute gradient w.r.t. token IDs
            optim_ids_onehot_grad = self.compute_token_gradient(optim_ids)

            with torch.no_grad():
                # Sample new candidate sequences from the gradient
                sampled_ids = sample_ids_from_grad(
                    optim_ids.squeeze(0),
                    optim_ids_onehot_grad.squeeze(0),
                    config.search_width,
                    config.topk,
                    config.n_replace,
                    not_allowed_ids=self.not_allowed_ids,
                )

                # Optional filter to ensure decode-then-encode is stable
                if config.filter_ids:
                    sampled_ids = filter_ids(sampled_ids, processor.tokenizer)

                # Evaluate the cross-entropy loss for all candidate prompts
                new_search_width = sampled_ids.shape[0]
                batch_size = new_search_width if config.batch_size is None else config.batch_size

                # Construct embeddings for each candidate
                if self.prefix_cache:
                    input_embeds = torch.cat([
                        embedding_layer(sampled_ids),
                        after_embeds.repeat(new_search_width, 1, 1),
                        target_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)
                else:
                    input_embeds = torch.cat([
                        before_embeds.repeat(new_search_width, 1, 1),
                        embedding_layer(sampled_ids),
                        after_embeds.repeat(new_search_width, 1, 1),
                        target_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)

                # We'll compute the candidate losses via single-image loops
                loss = find_executable_batch_size(self.compute_candidates_loss, batch_size)(input_embeds)

                # Choose the best candidate based on minimal loss
                current_loss = loss.min().item()
                best_idx = loss.argmin()
                optim_ids = sampled_ids[best_idx].unsqueeze(0)

                # Check if we've improved the best loss
                if current_loss < best_loss_so_far:
                    best_loss_so_far = current_loss
                    no_improvement_count = 0
                    logger.info(f"Step {step}: New best loss: {best_loss_so_far:.4f}")
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= max_patience:
                        logger.info(f"Early stopping: no improvement for {max_patience} steps. Best loss: {best_loss_so_far:.4f}")
                        break

                # Early stop info
                perfect_match_optim_str = ''
                if self.stop_flag:
                    perfect_match_optim_ids = optim_ids
                    perfect_match_optim_str = processor.tokenizer.batch_decode(optim_ids)[0]

                # Update buffer
                losses.append(current_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)

                # Use the buffer's best so far
                optim_ids = buffer.get_best_ids()
                optim_str = processor.tokenizer.batch_decode(optim_ids)[0]
                optim_strings.append(optim_str)
                buffer_msg = buffer.log_buffer(tokenizer=processor.tokenizer)
                logger.info(buffer_msg)

                if self.stop_flag:
                    logger.info("Early stopping due to finding a perfect match for all images.")
                    break

        min_loss_index = losses.index(min(losses))
        total_time = time.time() - start_time

        if not perfect_match_optim_str:
            # No exact match found; use the best overall from buffer
            perfect_match_optim_ids = buffer.get_best_ids()
            perfect_match_optim_str = processor.tokenizer.batch_decode(perfect_match_optim_ids)[0]
            logger.info(f"No perfect match found; using best-loss prompt: {perfect_match_optim_str}")

        result = GCGResult(
            best_loss=losses[min_loss_index],
            best_strings=optim_strings[min_loss_index],
            losses=losses,
            strings=optim_strings,
            perfect_match_optim_strs=perfect_match_optim_str,
            perfect_match_before_ids=self.before_ids,
            perfect_match_optim_ids=perfect_match_optim_ids,
            perfect_match_after_ids=self.after_ids,
            num_steps_taken=steps_taken,
            total_time=total_time,
            init_str_length=init_str_length
        )
        return result

    def init_buffer(self) -> AttackBuffer:
        """
        Initializes the candidate strings (IDs) buffer, 
        computing their losses and storing them sorted.
        """
        logger.info(f"Initializing attack buffer of size {self.config.buffer_size}...")
        buffer = AttackBuffer(self.config.buffer_size)

        processor, config = self.processor, self.config
        model = self.model

        # Convert user-provided `optim_str_init` into token IDs
        if isinstance(config.optim_str_init, str):
            init_optim_ids = processor.tokenizer(
                config.optim_str_init, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device)
            if config.buffer_size > 1:
                # fill up the rest with random picks from INIT_CHARS
                init_buffer_ids = processor.tokenizer(
                    INIT_CHARS, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].squeeze().to(model.device)
                init_indices = torch.randint(
                    0, init_buffer_ids.shape[0],
                    (config.buffer_size - 1, init_optim_ids.shape[1])
                )
                init_buffer_ids = torch.cat([init_optim_ids, init_buffer_ids[init_indices]], dim=0)
            else:
                init_buffer_ids = init_optim_ids
        else:
            # user passed a list of strings
            if len(config.optim_str_init) != config.buffer_size:
                logger.warning(
                    f"Using {len(config.optim_str_init)} init strings but buffer size is {config.buffer_size}"
                )
            try:
                init_buffer_ids = processor.tokenizer(
                    config.optim_str_init, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(model.device)
            except ValueError:
                logger.error(
                    "Unable to create buffer. Ensure each init string has the same tokenize length."
                )
                raise

        true_buffer_size = max(1, config.buffer_size)

        # Construct embeddings for each candidate
        if self.prefix_cache:
            # skip multi-img prefix caching logic
            init_buffer_embeds = torch.cat([
                self.embedding_layer(init_buffer_ids),
                self.after_embeds.repeat(true_buffer_size, 1, 1),
                self.target_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)
        else:
            init_buffer_embeds = torch.cat([
                self.before_embeds.repeat(true_buffer_size, 1, 1),
                self.embedding_layer(init_buffer_ids),
                self.after_embeds.repeat(true_buffer_size, 1, 1),
                self.target_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)

        # Get initial losses
        init_buffer_losses = find_executable_batch_size(
            self.compute_candidates_loss, true_buffer_size
        )(init_buffer_embeds)

        # Populate
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])
        buffer_msg = buffer.log_buffer(tokenizer=processor.tokenizer)
        logger.info(buffer_msg)
        logger.info("Initialized attack buffer.")
        return buffer

    def compute_token_gradient(self, optim_ids: torch.Tensor) -> torch.Tensor:
        """
        Computes the gradient of the negative log-likelihood w.r.t. 
        the token IDs (one-hot) by doing single-image forward passes. 
        This ensures alignment with how `predict_action` is typically called.
        """
        model = self.model
        embedding_layer = self.embedding_layer

        # Build one-hot representation that we can backprop
        optim_ids_onehot = torch.nn.functional.one_hot(
            optim_ids, num_classes=embedding_layer.num_embeddings
        ).to(model.device, model.dtype)
        optim_ids_onehot.requires_grad_()

        n_imgs = self.pixel_values.shape[0]
        sum_loss = 0.0

        # Sum the NLL across all images
        for img_idx in range(n_imgs):
            single_image_pixels = self.pixel_values[img_idx : img_idx+1]
            # Convert one-hot to embeddings
            candidate_embeds = optim_ids_onehot @ embedding_layer.weight

            if self.prefix_cache:
                raise NotImplementedError(
                    "Prefix caching for multi-image single loops is not yet implemented."
                )

            # Build the prompt embeddings
            input_embeds = torch.cat([
                self.before_embeds, 
                candidate_embeds, 
                self.after_embeds, 
                self.target_embeds
            ], dim=1)

            # Forward pass
            outputs = model(inputs_embeds=input_embeds, pixel_values=single_image_pixels)
            logits = outputs.logits

            target_len = self.target_ids.shape[1]
            shift_logits = logits[..., -target_len-1:-1, :].contiguous()
            shift_labels = self.target_ids  # shape = (1, target_len)

            # Cross-entropy for this single image
            if self.config.use_mellowmax:
                label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                # shape => (1, target_len)
                loss_here = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1).mean()
            else:
                loss_here = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="mean"
                )

            sum_loss += loss_here

        # Average across images
        final_loss = sum_loss / n_imgs

        # Backprop to get one-hot gradient
        (optim_ids_onehot_grad,) = torch.autograd.grad(final_loss, optim_ids_onehot, retain_graph=False)
        return optim_ids_onehot_grad

    def compute_candidates_loss(self, search_batch_size: int, input_embeds: Tensor) -> Tensor:
        """Computes the GCG loss on all candidate token id sequences."""
        all_loss = []
        prefix_cache_batch = []
        n_imgs = self.pixel_values.shape[0]  # Get number of images

        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i:i+search_batch_size]
                current_batch_size = input_embeds_batch.shape[0]

                # Process each image separately and average the losses
                batch_losses = []
                for img_idx in range(n_imgs):
                    pixel_values_batch = self.pixel_values[img_idx : img_idx + 1]
                    expand_dims = [current_batch_size] + [-1] * (pixel_values_batch.dim() - 1)
                    pixel_values_batch = pixel_values_batch.expand(*expand_dims)
                    
                    if self.prefix_cache:
                        if not prefix_cache_batch or current_batch_size != search_batch_size:
                            prefix_cache_batch = [[x.expand(current_batch_size, -1, -1, -1) for x in self.prefix_cache[i]] for i in range(len(self.prefix_cache))]
                        outputs = self.model(inputs_embeds=input_embeds_batch, pixel_values=pixel_values_batch, past_key_values=prefix_cache_batch)
                    else:
                        outputs = self.model(inputs_embeds=input_embeds_batch, pixel_values=pixel_values_batch)

                    logits = outputs.logits
                    target_len = self.target_ids.shape[1]
                    shift_logits = logits[..., -target_len-1:-1, :].contiguous()
                    shift_labels = self.target_ids.repeat(current_batch_size, 1)
                    
                    if self.config.use_mellowmax:
                        label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                        loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
                    else: 
                        loss = torch.nn.functional.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)), 
                            shift_labels.view(-1), 
                            reduction="none"
                        ).view(current_batch_size, -1).mean(dim=-1)
                    
                    batch_losses.append(loss)

                    del outputs
                    gc.collect()
                    torch.cuda.empty_cache()

                # Average losses across all images
                avg_loss = torch.stack(batch_losses).mean(dim=0)
                all_loss.append(avg_loss)

                # Early stopping checks for perfect match
                if self.config.early_stop:
                    best_sequence_idx = avg_loss.argmin()
                    best_loss = avg_loss[best_sequence_idx]
                    
                    perfect_match = True
                    for img_idx in range(n_imgs):
                        pixel_values_batch = self.pixel_values[img_idx : img_idx + 1]
                        expand_dims = [1] + [-1] * (pixel_values_batch.dim() - 1) 
                        pixel_values_batch = pixel_values_batch.expand(*expand_dims)

                        best_text = input_embeds_batch[best_sequence_idx : best_sequence_idx + 1]
                        outputs = self.model(inputs_embeds=best_text, pixel_values=pixel_values_batch)
                        logits = outputs.logits
                        
                        shift_logits = logits[..., -target_len-1:-1, :].contiguous()
                        predicted_tokens = torch.argmax(shift_logits, dim=-1)[0]
                        target_tokens = self.target_ids[0]
                        if not torch.equal(predicted_tokens, target_tokens):
                            perfect_match = False
                            break

                    if perfect_match:
                        if avg_loss.min() == torch.cat(all_loss, dim=0).min():
                            print(f"Perfect match predicted tokens: {predicted_tokens}")
                            print(f"Perfect match target tokens: {target_tokens}")
                            self.stop_flag = True
                            logger.info("Early stopping: perfect match found with the minimum loss across all images.")
                            break

        return torch.cat(all_loss, dim=0)


def robo_run(   
    model: OpenVLAForActionPrediction,
    processor: transformers.AutoProcessor,
    target: str,
    pixel_values: torch.FloatTensor,
    instruction: Optional[str] = None,
    config: Optional[GCGConfig] = None, 
) -> GCGResult:
    """
    A simple wrapper around the multi-image GCG run.
    Pass in any number of images (1..N) in pixel_values.
    """
    if config is None:
        config = GCGConfig()
    
    logger.setLevel(getattr(logging, config.verbosity))
    
    gcg = ROBOGCG(model, processor, config, pixel_values=pixel_values)
    result = gcg.run(target, instruction)
    return result
