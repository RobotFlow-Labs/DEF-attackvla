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
from transformers import AutoModel
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
        model: AutoModel, 
        processor: transformers.AutoProcessor,
        config: GCGConfig,
        image,
    ):
        self.model = model
        self.processor = processor
        self.image = image  # shape = (n_imgs, C, H, W)
        self.config = config

        self.embedding_layer = model.language_model.get_input_embeddings()
        self.not_allowed_ids = None if config.allow_non_ascii else get_nonascii_toks(
            self.processor.tokenizer, device=model.device
        )
        self.stop_flag = False

        if model.dtype in (torch.float32, torch.float64):
            logger.warning(
                f"Model is in {model.dtype}. Consider a lower precision (e.g. bfloat16) for faster optimization."
            )
        if model.device == torch.device("cpu"):
            logger.warning("Model is on CPU. Consider using a GPU or other accelerator for speed.")
    
    def run(self, action, instruction: Optional[str] = None) -> GCGResult:
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
            # assert instruction is not None, "Instruction must be provided if optimizing as a suffix"
            # if config.use_trace:
            #     template = "In: You are given two images: one with the original robot observation, and another one marked with historical traces of the robot end effector and moving objects, separated by a special separator token. What action should the robot take to {instruction} {optim_str}?\nOut: "
            # else:
            #     template = "In: What action should the robot take to {instruction} {optim_str}?\nOut: "
            #     template = template.replace("{instruction}", instruction)
            raise NotImplementedError("Suffix is not implemented.")
        else:
            prompt = f"What action should the robot take to {config.optim_str_init}?"
        # logger.info(f"Template used: {template}")
        # before_str, after_str = template.split("{optim_str}")
        # # Tokenize the static portions (before/after) and the target
        # before_ids = processor.tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(model.device) # remove the last token since spaces get encoded weird
        # after_ids = processor.tokenizer([after_str], add_special_tokens=False, padding=False, return_tensors="pt")["input_ids"].to(model.device)
        # after_ids[0][-1] = 29871
        # target_ids = processor.tokenizer([target], add_special_tokens=False, padding=False, return_tensors="pt")["input_ids"].to(model.device)[:, -7:]
        # # Store them for later
        # self.before_ids = before_ids
        # self.after_ids = after_ids
        # self.target_ids = target_ids
        # logger.info(f"Before IDs: {before_ids}")
        # logger.info(f"After IDs: {after_ids}")
        # logger.info(f"Target IDs: {target_ids}")
        # Embed static tokens
        logger.info(f"Prompt used: {prompt}")
        ori_inputs = processor(images=self.image,text=prompt,return_tensors="pt",suffix_actions=action,padding=True).to(torch.bfloat16).to(model.device)
        embedding_layer = self.embedding_layer
        ori_embeds = embedding_layer(ori_inputs["input_ids"]).detach()
        optim_ids = ori_inputs["input_ids"][:,-26:-6] ##ids consist of 256 images token + 1 bos token + 7 pre_prompt token + 20 optim_str_token + 1 ? token + 1 \n token + 3 action tokens + 1 eos token = 290 tokens
        # tmp=ori_inputs["input_ids"][:,-4:]
        # print(f"ori_embeds and its values:{ori_embeds.shape},{ori_embeds} action_ids:{tmp}")
        # print(f"mask shape and values:{mask.shape} {mask}")
        losses, best_strings = [], []
        init_str_length = (
            len(config.optim_str_init)
            if isinstance(config.optim_str_init, str)
            else len(config.optim_str_init[0])
        )
        steps_taken = 0

        # Initialize best loss tracking
        best_loss_so_far = float('inf')
        global_best_ids = None
        no_improvement_count = 0
        max_patience = 1000

        for step in tqdm(range(config.num_steps)):
            steps_taken = step + 1
            # Compute gradient w.r.t. token IDs

            ## 将当前optim_ids one hot 化后对target_ids求ce，得到每一个token上的梯度作为候选
            optim_ids_onehot_grad = self.compute_token_gradient(ori_inputs)
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

                # Construct embeddings for each candidate by replacing optim_ids positions
                # Start with repeated full sequence embeddings
                input_embeds = ori_embeds.repeat(new_search_width, 1, 1)  # [new_search_width, seq_len, embed_dim]
                print(f"sampled_ids shape and value {sampled_ids.shape},{sampled_ids[0]}")
                # Replace the optim_ids positions with new sampled embeddings
                optim_embeds = embedding_layer(sampled_ids)  # [new_search_width, optim_len, embed_dim]
                input_embeds[:, -26:-6, :] = optim_embeds  # Replace positions [-26:-6] with new embeddings
                
                # Create corresponding input_ids for each candidate
                candidate_input_ids = ori_inputs["input_ids"].repeat(new_search_width, 1)  # [new_search_width, seq_len]
                candidate_input_ids[:, -26:-6] = sampled_ids  # Replace optim_ids positions

                # We'll compute the candidate losses via single-image loops
                loss = find_executable_batch_size(self.compute_candidates_loss, batch_size)(input_embeds,ori_inputs,candidate_input_ids)

                # Choose the best candidate based on minimal loss
                current_loss = loss.min().item()
                best_idx = loss.argmin()
                best_ids = candidate_input_ids[best_idx].unsqueeze(0)

                # Check if we've improved the best loss
                if current_loss < best_loss_so_far:
                    best_loss_so_far = current_loss
                    global_best_ids = best_ids.clone()  # 保存全局最优
                    no_improvement_count = 0
                    logger.info(f"Step {step}: New best loss: {best_loss_so_far:.4f}")
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= max_patience:
                        logger.info(f"Early stopping: no improvement for {max_patience} steps. Best loss: {best_loss_so_far:.4f}")
                        break

                # Early stop info
                perfect_match_strs = ''
                if self.stop_flag:
                    perfect_match_ids = best_ids
                    perfect_match_strs = processor.tokenizer.batch_decode(best_ids)[0]

                # Update state
                losses.append(current_loss)
                # 使用全局最优进行更新
                if global_best_ids is not None:
                    best_str = processor.tokenizer.batch_decode(global_best_ids)[0]
                    best_strings.append(best_str)
                    ori_inputs['input_ids'] = global_best_ids
                else:
                    # 第一次迭代，使用当前最优
                    best_str = processor.tokenizer.batch_decode(best_ids)[0]
                    best_strings.append(best_str)
                    ori_inputs['input_ids'] = best_ids

                if self.stop_flag:
                    logger.info("Early stopping due to finding a perfect match for all images.")
                    break

        min_loss_index = losses.index(min(losses))
        total_time = time.time() - start_time

        if not perfect_match_strs:
            # No exact match found; use the global best
            perfect_match_ids = global_best_ids if global_best_ids is not None else best_ids
            perfect_match_strs = processor.tokenizer.batch_decode(perfect_match_ids)[0]
            logger.info(f"No perfect match found; using best-loss prompt: {perfect_match_strs}")

        result = GCGResult(
            best_loss=losses[min_loss_index],
            best_strings=best_strings[min_loss_index],
            losses=losses,
            strings=best_strings,
            perfect_match_strs=perfect_match_strs,
            perfect_match_ids=perfect_match_ids,
            num_steps_taken=steps_taken,
            total_time=total_time,
            init_str_length=init_str_length
        )
        return result


    def compute_token_gradient(self, ori_inputs) -> torch.Tensor:
        """
        Computes the gradient of the negative log-likelihood w.r.t. 
        the token IDs (one-hot) by doing single-image forward passes. 
        This ensures alignment with how `predict_action` is typically called.
        """
        model = self.model
        embedding_layer = self.embedding_layer
        
        input_ids_onehot = torch.nn.functional.one_hot(
            ori_inputs['input_ids'], num_classes=embedding_layer.num_embeddings
        ).to(model.device, model.dtype)
        input_ids_onehot.requires_grad_()               
        n_imgs = ori_inputs['pixel_values'].shape[0]
        sum_loss = 0.0
        # Sum the NLL across all images
        for img_idx in range(n_imgs):
            # Convert input_ids to embeddings
            input_embeds = input_ids_onehot @ embedding_layer.weight
            model_input = {
                "input_ids": ori_inputs['input_ids'],
                "inputs_embeds": input_embeds,
                "attention_mask": ori_inputs['attention_mask'],
                "pixel_values": ori_inputs['pixel_values'],
                "intrinsic": ori_inputs['intrinsic']
            }
            
            # Forward pass
            outputs = model(**model_input, return_dict=True)
            logits = outputs.logits
            # print(f"logits shape:{logits.shape}")
            
            # Get target tokens (last few tokens corresponding to action)
            target_tokens = ori_inputs['input_ids'][:, -4:-1]  # Action tokens are at positions -4, -3, -2
            # print(f"target_tokens: {target_tokens}")
            # Compute loss
            target_len = target_tokens.shape[1]  # target_len = 3
            # logits[i] predicts input_ids[i+1], so we need logits at positions -5, -4, -3
            shift_logits = logits[..., -target_len-2:-2, :].contiguous()  # logits[..., -5:-2, :]
            shift_labels = target_tokens
            
            # Cross-entropy loss
            if self.config.use_mellowmax:
                # label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                # loss_here = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1).mean()
                raise NotImplementedError("Mellowmax is not implemented.")
            else:
                loss_here = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="mean"
                )
            
            sum_loss += loss_here

        # Average across images
        final_loss = sum_loss / n_imgs
        # Backprop to get gradients
        (input_ids_onehot_grad,) = torch.autograd.grad(final_loss, input_ids_onehot, retain_graph=False)
        return input_ids_onehot_grad[:,-26:-6,:]

    def compute_candidates_loss(self, search_batch_size: int, input_embeds: Tensor, ori_inputs, candidate_input_ids) -> Tensor:
        """Computes the GCG loss on all candidate token id sequences."""
        all_loss = []
        target_ids = ori_inputs["input_ids"][:,-4:-1]
        pixel_values = ori_inputs["pixel_values"]
        n_imgs = pixel_values.shape[0]
        
        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i:i+search_batch_size]
                current_batch_size = input_embeds_batch.shape[0]

                # Process each image separately and average the losses
                batch_losses = []
                for img_idx in range(n_imgs):
                    pixel_values_batch = pixel_values[img_idx : img_idx + 1]
                    expand_dims = [current_batch_size] + [-1] * (pixel_values_batch.dim() - 1)
                    pixel_values_batch = pixel_values_batch.expand(*expand_dims)
                    # Handle intrinsic parameter correctly
                    if 'intrinsic' in ori_inputs:
                        intrinsic_batch = ori_inputs['intrinsic'].repeat(current_batch_size, 1, 1)  # [batch_size, 3, 3]
                    else:
                        intrinsic_batch = None
                    
                    model_inputs = {
                        "input_ids": candidate_input_ids[i:i+search_batch_size],
                        "inputs_embeds": input_embeds_batch.clone(),
                        "attention_mask": ori_inputs['attention_mask'].repeat(current_batch_size, 1),
                        "pixel_values": pixel_values_batch,
                    }
                    
                    if intrinsic_batch is not None:
                        model_inputs["intrinsic"] = intrinsic_batch
                    outputs = self.model(**model_inputs,return_dict=True)

                    logits = outputs.logits
                    target_len = target_ids.shape[1]
                    shift_logits = logits[..., -target_len-2:-2, :].contiguous()
                    shift_labels = target_ids.repeat(current_batch_size, 1)
                    
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
                        pixel_values_batch = pixel_values[img_idx : img_idx + 1]
                        expand_dims = [1] + [-1] * (pixel_values_batch.dim() - 1) 
                        pixel_values_batch = pixel_values_batch.expand(*expand_dims)

                        best_text = input_embeds_batch[best_sequence_idx : best_sequence_idx + 1]
                        # Get the corresponding input_ids for the best sequence
                        best_input_ids = candidate_input_ids[best_sequence_idx:best_sequence_idx+1]
                        
                        # Handle intrinsic parameter correctly
                        if 'intrinsic' in ori_inputs:
                            intrinsic_batch = ori_inputs['intrinsic'].repeat(1, 1, 1)  # [1, 3, 3]
                        else:
                            intrinsic_batch = None
                        
                        model_inputs = {
                            "input_ids": best_input_ids,
                            "inputs_embeds": best_text,
                            "attention_mask": ori_inputs['attention_mask'],
                            "pixel_values": pixel_values_batch,
                            "intrinsic":ori_inputs['intrinsic']
                        }
                        outputs = self.model(**model_inputs,return_dict=True)
                        logits = outputs.logits
                        
                        shift_logits = logits[..., -target_len-2:-2, :].contiguous()
                        predicted_tokens = torch.argmax(shift_logits, dim=-1)[0]
                        target_tokens = target_ids[0]
                        # logger.info(f"predicted_tokens: {predicted_tokens} target_tokens: {target_tokens}")
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
    def make_mask(self, ori_embeds, ori_inputs, model, processor):
        pad_token_id = processor.tokenizer.pad_token_id
        pad_counts = (ori_inputs["input_ids"] == pad_token_id).sum(dim=1)
        # print(f"pad_counts {pad_counts}")
        num_image_tokens = model.config.text_config.num_image_tokens
        prefix = pad_counts + num_image_tokens + 8 # 8 is the length of "bos What action should the robot take to"
        # print(f"{prefix+2} tokens do not need grad!!")
        mask = torch.ones_like(ori_embeds)
        for b in range(mask.shape[0]):
            mask[b, :prefix[b], :] = 0
        mask[:, -6:, :] = 0 ##?+\n+action+eos
        return mask
def robo_run(   
    model: AutoModel,
    processor: transformers.AutoProcessor,
    # target: str,
    action,
    image,
    # pixel_values: torch.FloatTensor,
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
    
    gcg = ROBOGCG(model, processor, config, image=image)
    result = gcg.run(action, instruction)
    return result
