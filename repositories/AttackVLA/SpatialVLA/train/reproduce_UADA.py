import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional, List
import json
import torch
import torch.distributed as dist
from train.dist_utils import init_dist
from train.monkey_patch import (
    concat_pad_data_collator,
    SaveProcessorCallback,
    replace_train_dataloader,
    replace_train_sampler,
)
from train.monkey_patch_UADA import PatchTrainer
import transformers
from transformers import (
    HfArgumentParser,
    set_seed,
    TrainingArguments,
)
from transformers.utils.logging import (
    enable_default_handler,
    enable_explicit_format,
    set_verbosity,
)
from data.dataset_UADA import build_datasets
from model import (
    SpatialVLAConfig,
    SpatialVLAForConditionalGeneration,
    SpatialActionTokenizer,
    SpatialVLAProcessorUADA,
)
replace_train_dataloader()
replace_train_sampler()

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: Optional[str] = field(default=None,
        metadata={"help": "Path to pretrained model or identifier for resume training."},
    )
    flash_attn: bool = field(
        default=True,
        metadata={"help": "Set to True to use Flash Attention 2.0."},
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_root_dir: Optional[str] = field(
        default="datasets/open-x-embodiment",
        metadata={"help": "The root directory of the dataset. Default is `data`."},
    )
    data_mix: Optional[str] = field(
        default="bridge",
        metadata={"help": "The name of the dataset mixture. Default is `bridge`."},
    )
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={"help": "The maximum total input sequence length after tokenization. "},
    )
    shuffle_buffer_size: Optional[int] = field(
        default=1000_000,
        metadata={"help": "The shuffle buffer size for the dataset. Default is 1000000."},
    )
    tsfm_thread_muti: Optional[int] = field(
        default=1,
        metadata={"help": "The threads number of rlds transfom. Default is 1."},
    )
    read_thread_muti: Optional[int] = field(
        default=1,
        metadata={"help": "The threads number of rlds reader. Default is 1."},
    )
    obs_backward_steps: Optional[int] = field(
        default=0,
        metadata={"help": "Number of backward steps in observation. 0 indicates current"},
    )
    obs_backward_delta: Optional[int] = field(
        default=1, metadata={"help": "Backward delta in observation."}
    )
    action_forward_steps: Optional[int] = field(
        default=0,
        metadata={"help": "Number of forward steps in action. 0 indicates current"},
    )
    fix_raw_length: Optional[int] = field(
        default=None, metadata={"help": "fix the iterable dataset iter length."}
    )
    use_raw_dataloader: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use raw dataloader"}
    )
    maskidx: Optional[List[int]] = field(
        default_factory=lambda: [0],
        metadata={
            "help": "Indices to supervise among {0:translation,1:rotation,2:gripper}, comma-separated (e.g., '0,1')."
        },
    )
    innerloop: Optional[int] = field(
        default=50,
        metadata={
            "help": "Number of inner loop iterations for patch training."
        },
    )

def main():
    launcher = os.environ.get("LAUNCHER", "slurm")
    init_dist(launcher=launcher, backend="nccl")
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log: transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint and eventually continue from last checkpoint.

    set_seed(training_args.seed)

    # 1. initializing models and load tokenizer
    _processor = SpatialVLAProcessorUADA.from_pretrained(model_args.model_name_or_path, local_files_only=True)
    tokenizer = _processor.tokenizer
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    
    logger.info("Loading SpatialVLA Model...")
    config = SpatialVLAConfig.from_pretrained(model_args.model_name_or_path, torch_dtype=torch_dtype, local_files_only=True)
    model = SpatialVLAForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        local_files_only=True
    )
    # Freeze all model parameters: patch-only training should not update model weights
    for param in model.parameters():
        param.requires_grad = False
    # Ensure gradients can flow through the vision backbone to the input patch if the model supports this flag
    if hasattr(model, "vision_backbone_requires_grad"):
        model.vision_backbone_requires_grad = True
    if model_args.flash_attn:
        model.language_model.config._attn_implementation = model.config.text_config._attn_implementation_internal = "flash_attention_2"
        model.vision_tower.config._attn_implementation = model.config.vision_config._attn_implementation_internal = "flash_attention_2"

    # 2. build datasets
    train_dataset, eval_dataset = build_datasets(
        data_args,
        training_args.output_dir,
        vla_processor=None,
    )

    # 3. build action tokenizer from current project
    action_tokenizer = SpatialActionTokenizer(
        tokenizer,
        num_bins=_processor.action_config["num_bins"],
        bin_policy=_processor.action_tokenizer.bin_policy,
        use_spherical=_processor.action_config["use_spherical"],
        min_sigma=_processor.action_config.get("min_sigma", 0.0),
    )

    model=model.eval()
    # overwrite attributes
    model.action_token_begin_idx = model.config.action_token_begin_idx = action_tokenizer.action_token_begin_idx
    model.vision_tower.gradient_checkpointing = True

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            # print(f"module name:{name},requires_grad:{param.requires_grad}")
            if param.requires_grad: logger.info(name)
    set_seed(training_args.seed)
    # Skip auto-class registration in patch training

    # build processor
    statistic = train_dataset.ds_stats_pc
    _processor.statistics.update(statistic)
    processor = SpatialVLAProcessorUADA(
        image_processor=_processor.image_processor,
        tokenizer=tokenizer,
        statistics=_processor.statistics,
        bin_policy=action_tokenizer.bin_policy,
        intrinsic_config=_processor.intrinsic_config,
        action_config=_processor.action_config,
        num_obs_steps=data_args.obs_backward_steps + 1,
        obs_delta=data_args.obs_backward_delta,
        action_chunk_size=data_args.action_forward_steps + 1,
    )

    model.action_tokenizer = action_tokenizer
    train_dataset.vla_processor = processor

    trainer = PatchTrainer(
        innerloop=data_args.innerloop,
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=concat_pad_data_collator,
        callbacks=[SaveProcessorCallback(processor=processor)],
    )

    if training_args.do_train:
        train_result = trainer.train()
        # trainer.save_model()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

if __name__ == "__main__":
    main()