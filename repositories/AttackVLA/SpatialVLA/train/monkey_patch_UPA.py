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

logger = logging.get_logger(__name__)

IGNORE_INDEX = -100

class PatchTrainer(Trainer):
    def __init__(self,suite="error",innerloop=50, alpha=0.8, belta=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Optimize only the adversarial patch
        self.patch = torch.nn.Parameter(torch.rand([3,50,50]).to(torch.device('cuda')), requires_grad=True)
        self.patch_save_step=2000
        self.innerloop = innerloop
        self.alpha = alpha
        self.belta = belta
        self.suite=suite
        print(f"UPA parameters - alpha: {self.alpha}, belta: {self.belta}")
        
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
        patch_save_dir=f"LIBERO/ad_patches/UPA/{self.suite}"
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
        UPA weighted loss for SpatialVLA - focuses only on translation (x,y,z).
        Adapted from OpenVLA UPA implementation.
        """
        import torch.nn.functional as F
        
        # Align logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()  # (bs, seq-1, voc)
        shift_labels = labels[..., 1:].contiguous()      # (bs, seq-1)
        # Get translation token range
        t_start = model.action_tokenizer.translation_tokenizer.token_start_idx
        t_end = model.action_tokenizer.translation_tokenizer.token_end_idx
        t_bins = t_end - t_start + 1  # Number of translation bins
        
        # Create action mask for translation tokens only
        action_mask = (shift_labels >= t_start) & (shift_labels <= t_end)
        if action_mask.sum() == 0:
            # No translation tokens found, return zero loss
            return torch.tensor(0.0, device=logits.device, requires_grad=True), 0.0, 0.0
        
        # Extract translation logits and labels
        temp_label = shift_labels[action_mask]  # (num_translation_tokens,)
        relevant_logits = shift_logits[action_mask]  # (num_translation_tokens, voc)
        temp_logits = relevant_logits[:, t_start:t_end+1]  # (num_translation_tokens, t_bins)
        # Create reweighting vector to encourage higher token IDs
        reweigh = torch.arange(1, t_bins + 1, device=logits.device)  # [1, 2, 3, ..., t_bins]
        
        # Compute softmax probabilities
        temp_prob = F.softmax(temp_logits, dim=-1)  # (num_translation_tokens, t_bins)
        # Compute reweighted probabilities (expected token ID)
        reweighted_prob = (temp_prob * reweigh).sum(dim=-1)  # (num_translation_tokens,)
        # Convert to continuous values in [0, 1] range (matching original UPA)
        xyz_reweighted = (reweighted_prob - 1) / (t_bins - 1)  # Map [1, t_bins] to [0, 1]
        
        # Convert ground truth token IDs to continuous values
        xyz_label = (temp_label - t_start) / (t_bins - 1)  # Map [t_start, t_end] to [0, 1]
        
        cosine_sim = F.cosine_similarity(xyz_reweighted, xyz_label, dim=0)
        angle_loss = (cosine_sim + 1).mean()
        distance_loss = 1/(torch.norm(xyz_reweighted - xyz_label, p=2, dim=0)+1e-3)

        total_loss = self.alpha*angle_loss + self.belta*distance_loss
        return total_loss, angle_loss.item(), distance_loss.item()


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
        # print(f"inputs keys(): {inputs.keys()}")
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
            # Use UPA weighted_loss for translation attack
            loss, angle_loss, distance_loss = self.weighted_loss(outputs["logits"], inputs["labels"], model)
        else:
            raise ValueError(
                "UPA training requires 'labels' in inputs for weighted loss computation. "
                "Please ensure your dataset provides labels for action tokens."
            )
            
        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        with torch.no_grad():
            logits = outputs["logits"]  # (bs, seq, voc)
            labels = inputs["labels"]  # (bs, seq)
            shift_logits = logits[..., :-1, :].argmax(-1).contiguous()
            shift_labels = labels[..., 1:].contiguous()

            mask = (shift_labels >= model.action_tokenizer.translation_tokenizer.token_start_idx) & (
                shift_labels <= model.action_tokenizer.gripper_tokenizer.token_end_idx
            )
            gt_action_ids, pred_action_ids = shift_labels[mask], shift_logits[mask]
            correct_preds = gt_action_ids == pred_action_ids
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # NOTE: acc of translation, rotation and gripper
            token_start_idx, token_end_idx = (
                model.action_tokenizer.translation_tokenizer.token_start_idx,
                model.action_tokenizer.translation_tokenizer.token_end_idx,
            )
            translation_mask = (gt_action_ids >= token_start_idx) & (gt_action_ids <= token_end_idx)

            translation_gt_action_ids, translation_pred_action_ids = gt_action_ids[translation_mask], pred_action_ids[translation_mask]

            translation_correct_preds = translation_gt_action_ids == translation_pred_action_ids

            translation_action_accuracy = translation_correct_preds.sum().float() / translation_mask.sum().float()
            
            info={
                    "accuracy": action_accuracy.item(),
                    "translation_accuracy": translation_action_accuracy.item(),
                    "angle_loss": angle_loss,
                    "distance_loss": distance_loss,
                }
        return (loss, outputs ,info) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Custom training step with inner loop for TMA patch training.
        For each batch, we train the patch multiple times (innerloop) on the same data.
        """
        original_inputs = inputs.copy()
        
        for i in range(self.innerloop):
            inputs = original_inputs.copy()            
            loss, outputs,info = self.compute_loss(model, inputs, return_outputs=True)
            
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
        self.log(info)
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
    