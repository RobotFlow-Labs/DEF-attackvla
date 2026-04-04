import os
import torch
import numpy as np
import torch.nn as nn
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
import torch.nn.functional as F

logger = logging.get_logger(__name__)

IGNORE_INDEX = -100

class PatchTrainer(Trainer):
    def __init__(self,suite="error",innerloop=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Optimize only the adversarial patch
        self.patch = torch.nn.Parameter(torch.rand([3,50,50]).to(torch.device('cuda')), requires_grad=True)
        self.patch_save_step=2000
        self.innerloop = innerloop
        self.suite=suite
        # TMA transform parameters
        self.angle = 30
        self.shx = 0.2
        self.shy = 0.2
        self.device = torch.device('cuda')
        
        # Normalization parameters (following SpatialVLA approach)
        # SpatialVLA uses (0.5, 0.5, 0.5) for both mean and std
        self.mean = torch.tensor([0.5, 0.5, 0.5]).to(self.device)
        self.std = torch.tensor([0.5, 0.5, 0.5]).to(self.device)

    def save_patch(self,step):
        patch_save_dir=f"LIBERO/ad_patch/UADA/{self.suite}"
        os.makedirs(patch_save_dir, exist_ok=True)
        patch_save_file=os.path.join(patch_save_dir,f"patch_{step}.pt")
        torch.save(self.patch.detach().cpu(),patch_save_file)
        print(f"patch saved to {patch_save_file}")

    def normalize(self, images, mean, std):
        """Normalize images using mean and std"""
        images = images - mean[None, :, None, None]
        images = images / std[None, :, None, None]
        return images

    def rotation_matrix(self, theta):
        theta = np.deg2rad(theta)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        return np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ], dtype=np.float32)

    def shear_matrix(self, shx, shy):
        return np.array([
            [1, shx, 0],
            [shy, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)

    def apply_affine_transform(self, image, transform_matrix):
        if image.ndim == 4:
            image = image.squeeze(0)
        affine_matrix = transform_matrix[:2, :].unsqueeze(0)  # [1, 2, 3]
        
        grid = torch.nn.functional.affine_grid(affine_matrix, image.unsqueeze(0).size(), align_corners=False)
        transformed_image = torch.nn.functional.grid_sample(image.unsqueeze(0), grid, align_corners=False, padding_mode='border')
        
        return transformed_image.squeeze(0)

    def combined_transform_matrix(self):
        if np.random.rand() < 0.2:
            return torch.tensor(np.eye(3, dtype=np.float32))
        else:
            angle = np.random.uniform(-self.angle, self.angle)
            shx = np.random.uniform(-self.shx, self.shx)
            shy = np.random.uniform(-self.shy, self.shy)

            R = self.rotation_matrix(angle)
            S = self.shear_matrix(shx, shy)
            combined_matrix = np.dot(S, R)
            return torch.tensor(combined_matrix)

    def apply_patch_to_tensors(self, images, patch):
        """
        Apply adversarial patch directly to tensor images to preserve gradients.
        Following TMA approach: directly operate on tensors without PIL conversion.
        
        Args:
            images: List of PIL Images or single PIL Image
            patch: Adversarial patch tensor [3, H, W]
        
        Returns:
            Tensor of patched images [batch_size, 3, H, W]
        """
        if not isinstance(images, list):
            images = [images]
        
        patched_tensors = []
        for img in images:
            # Convert PIL to tensor for patch application
            img_tensor = transforms.ToTensor()(img).to(self.device)
            img_channels, img_height, img_width = img_tensor.shape
            
            # Create canvas for patch placement
            canvas = torch.ones(img_channels, img_height, img_width).to(self.device) * -100
            
            # Get patch dimensions
            patch_channels, patch_height, patch_width = patch.shape
            
            # Random patch placement
            max_x = img_width - patch_width
            max_y = img_height - patch_height
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            
            # Place patch on canvas
            canvas[:, y:y + patch_height, x:x + patch_width] = patch
            
            # Apply geometric transformation
            affline_matrix = self.combined_transform_matrix().to(self.device)
            canvas = self.apply_affine_transform(canvas, affline_matrix)
            patched_tensor = torch.where(canvas < -20, img_tensor, canvas)
            # Normalize the patched tensor using TMA approach
            patched_tensor = self.normalize(patched_tensor, self.mean, self.std)
            patched_tensors.append(patched_tensor.squeeze())
        
        # Stack tensors to create batch
        return torch.stack(patched_tensors, dim=0)

    def weighted_loss(self, logits, labels, model):
        """
        SpatialVLA version of weighted loss for UADA.
        Handles different bin sizes for translation, rotation, and gripper dimensions.
        """
        # Use same alignment logic as compute_loss
        shift_logits = logits[..., :-1, :].contiguous()  # (bs, seq-1, voc)
        shift_labels = labels[..., 1:].contiguous()      # (bs, seq-1)
        # Create action mask
        action_mask = (shift_labels >= model.action_tokenizer.translation_tokenizer.token_start_idx) & (
            shift_labels <= model.action_tokenizer.gripper_tokenizer.token_end_idx
        )
        
        # Get action logits and labels
        action_logits = shift_logits[action_mask]  # (N, voc)
        action_labels = shift_labels[action_mask]  # (N,)
        
        # Get token ranges for each component
        t_start = model.action_tokenizer.translation_tokenizer.token_start_idx
        t_end = model.action_tokenizer.translation_tokenizer.token_end_idx
        r_start = model.action_tokenizer.rotation_tokenizer.token_start_idx
        r_end = model.action_tokenizer.rotation_tokenizer.token_end_idx
        g_start = model.action_tokenizer.gripper_tokenizer.token_start_idx
        g_end = model.action_tokenizer.gripper_tokenizer.token_end_idx
        
        # Calculate bin sizes for each component
        t_bins = t_end - t_start + 1
        r_bins = r_end - r_start + 1
        g_bins = g_end - g_start + 1
        
        # Initialize reweighted probabilities and hard max labels
        reweighted_probs = []
        hard_max_labels = []
        
        # Process each component
        for comp_idx, (comp_start, comp_end, comp_bins) in enumerate([(t_start, t_end, t_bins), 
                                                                     (r_start, r_end, r_bins), 
                                                                     (g_start, g_end, g_bins)]):
            # Create mask for this component
            comp_mask = (action_labels >= comp_start) & (action_labels <= comp_end)
            if comp_mask.sum() == 0:
                continue
                
            # Extract logits for this component
            comp_logits = action_logits[comp_mask, comp_start:comp_end+1]  # (N, comp_bins)
            comp_labels = action_labels[comp_mask] - comp_start  # Convert to 0-based indexing
            
            # Create reweighting for this component
            reweigh = torch.arange(1, comp_bins + 1).to(logits.device) / comp_bins  # [1/comp_bins, 2/comp_bins, ..., 1]
            
            # Calculate reweighted probability
            temp_prob = F.softmax(comp_logits, dim=-1)
            reweighted_prob = (temp_prob * reweigh).sum(dim=-1)
            reweighted_probs.append(reweighted_prob)
            
            # Create hard max labels for this component
            comp_midpoint = comp_bins // 2
            hard_max_label = torch.where(
                comp_labels > comp_midpoint,
                torch.tensor(1.0).to(logits.device),  # High value for upper half
                torch.tensor(1.0 / comp_bins).to(logits.device)  # Low value for lower half
            )
            hard_max_labels.append(hard_max_label)
        
        # Concatenate all reweighted probabilities and hard max labels
        if reweighted_probs:
            all_reweighted_probs = torch.cat(reweighted_probs, dim=0)
            all_hard_max_labels = torch.cat(hard_max_labels, dim=0)
            
            # Calculate single MSE loss
            total_loss = F.mse_loss(5 * all_reweighted_probs.contiguous(), 5 * all_hard_max_labels.float().contiguous())
        else:
            total_loss = torch.tensor(0.0).to(logits.device)
        
        return total_loss


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # print("🔥 CUSTOM compute_loss called! 🔥")  # 调试信息
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        # Apply adversarial patch to raw images and preprocess
        raw_images = inputs["pixel_values"]

        # Apply patch directly to tensor to preserve gradients
        patched_tensors = self.apply_patch_to_tensors(raw_images, self.patch)

        # Convert to the format expected by the model
        inputs["pixel_values"] = patched_tensors
        
        # all tensor on the same device
        device = next(model.parameters()).device
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(device)        
        outputs = model(**inputs)
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if "labels" in inputs:
            # Use weighted_loss for UADA training
            loss = self.weighted_loss(outputs["logits"], inputs["labels"], model)
        else:
            raise ValueError(
                "UADA training requires 'labels' in inputs for weighted loss computation. "
                "Please ensure your dataset provides labels for action tokens."
            )
            
        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Custom training step with inner loop for TMA patch training.
        For each batch, we train the patch multiple times (innerloop) on the same data.
        """
        original_inputs = inputs.copy()
        
        for i in range(self.innerloop):
            inputs = original_inputs.copy()            
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            
            loss.backward()            
            self.optimizer.step()
            self.patch.data = torch.clamp(self.patch.data, 0, 1)
            self.optimizer.zero_grad()
            model.zero_grad()
            torch.cuda.empty_cache()
        # Save patch periodically
        if (self.state.global_step + 1) % self.patch_save_step == 0 or (self.state.global_step + 1)%500 ==0:
            self.save_patch(step=self.state.global_step)
            if (self.state.global_step + 1)==2000:
                exit()
        self.lr_scheduler.step()
        return loss
    
    def create_optimizer(self):
        # print("🔥 CUSTOM optimizer called! 🔥")  # 调试信息
        # Override to optimize only the patch (AdamW), following TMA style
        import transformers as _tf
        
        print(f"Creating optimizer with patch parameters: {self.patch.shape}, requires_grad: {self.patch.requires_grad}")
        self.optimizer = _tf.AdamW([self.patch], lr=self.args.learning_rate)
        
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        # print("🔥 CUSTOM scheduler called! 🔥")  # 调试信息
        # Cosine schedule with warmup, similar to TMA
        if self.lr_scheduler is None:
            import transformers as _tf
            opt = optimizer if optimizer is not None else self.optimizer
            # Derive warmup steps from args
            warmup_steps = getattr(self.args, "warmup_steps", 0)
            if warmup_steps == 0 and hasattr(self.args, "get_warmup_steps"):
                try:
                    warmup_steps = self.args.get_warmup_steps(num_training_steps)
                except Exception:
                    warmup_steps = 0
            self.lr_scheduler = _tf.get_cosine_schedule_with_warmup(
                optimizer=opt,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=0.5,
            )
        return self.lr_scheduler
    