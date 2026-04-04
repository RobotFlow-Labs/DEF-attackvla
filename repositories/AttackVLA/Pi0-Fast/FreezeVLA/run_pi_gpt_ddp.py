import os
import torch
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModel, AutoProcessor, AutoModelForVision2Seq
from utils_pi import *

def evaluate_multi_prompt_ref(end_token_id, dataloader, vla, processor, ref, args, rank, logger):
    if rank == 0:
        logger.info(f"Multi Prompt Attack ({len(ref)} refs) adversarial prediction:")
    multi_prompt_asr = 0
    local_count = 0
    for (texts, images, _) in dataloader:
        if isinstance(images[0], list):
            images = [img for sublist in images for img in sublist]

        mp_images = []
        for i in range(len(texts)):

            adv_image = multi_prompt_attack(end_token_id, vla, processor, images[i], ref, args.eps, args.alpha, args.image_step, random_start=True)
            mp_images.append(adv_image)
        mp_images = torch.cat(mp_images, dim=0)

        inputs_test = processor(images=images, text=texts, return_tensors="pt", padding=True)
        inputs_test["pixel_values"] = mp_images
        eos_count = predict_action_batch(end_token_id, vla.module, inputs_test.to(vla.module.device))

        multi_prompt_asr += eos_count
        local_count += len(texts)


    # Aggregate multi_prompt_asr from all distributed processes
    local_asr_tensor = torch.tensor(multi_prompt_asr, device='cuda')
    local_count_tensor = torch.tensor(local_count, device='cuda')
    dist.all_reduce(local_asr_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_count_tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        logger.info(f"Multi Prompt Attack ({len(ref)} refs) ASR: {local_asr_tensor.item()} / {local_count_tensor.item()}")

def evaluate_multi_prompt_attack(end_token_id, dataloader, vla, processor, num_ref, args, rank, logger):
    if rank == 0:
        logger.info(f"Multi Prompt Attack ({num_ref} refs) adversarial prediction:")
    multi_prompt_asr = 0
    local_count = 0
    for (texts, images, ref_prompts) in dataloader:
        if isinstance(images[0], list):
            images = [img for sublist in images for img in sublist]

        mp_images = []
        for i in range(len(texts)):
            prompt_list = random.sample(ref_prompts[i], num_ref)
            adv_image = multi_prompt_attack(end_token_id, vla, processor, images[i], prompt_list, args.eps, args.alpha, args.image_step, random_start=True)
            mp_images.append(adv_image)
        mp_images = torch.cat(mp_images, dim=0)

        inputs_test = processor(images=images, text=texts, return_tensors="pt", padding=True)
        inputs_test["pixel_values"] = mp_images

        eos_count = predict_action_batch(end_token_id, vla.module, inputs_test.to(vla.module.device))

        multi_prompt_asr += eos_count
        local_count += len(texts)

    # Aggregate multi_prompt_asr from all distributed processes
    local_asr_tensor = torch.tensor(multi_prompt_asr, device='cuda')
    local_count_tensor = torch.tensor(local_count, device='cuda')
    dist.all_reduce(local_asr_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_count_tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        logger.info(f"Multi Prompt Attack ({num_ref} refs) ASR: {local_asr_tensor.item()} / {local_count_tensor.item()}")

def evaluate_stop_action_ref(end_token_id, dataloader, vla, processor, ref, args, rank, logger):
    if rank == 0:
        logger.info(f"Stop Action Attack ({len(ref)} refs) adversarial prediction:")
    stop_action_asr = 0
    local_count = 0
    for (texts, images, _) in dataloader:
        if isinstance(images[0], list):
            images = [img for sublist in images for img in sublist]

        mp_images = []
        for i in range(len(texts)):
            adv_image = stop_action_token_attack(end_token_id, vla, processor, images[i], ref, args.text_step, args.eps, args.alpha, args.image_step, random_start=True)
            mp_images.append(adv_image)
        mp_images = torch.cat(mp_images, dim=0)

        inputs_test = processor(images=images, text=texts, return_tensors="pt", padding=True)
        inputs_test["pixel_values"] = mp_images

        eos_count = predict_action_batch(end_token_id, vla.module, inputs_test.to(vla.module.device))

        stop_action_asr += eos_count
        local_count += len(texts)

    local_asr_tensor = torch.tensor(stop_action_asr, device='cuda')
    local_count_tensor = torch.tensor(local_count, device='cuda')
    dist.all_reduce(local_asr_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_count_tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        logger.info(f"StopAction Ours ({len(ref)} refs) ASR: {local_asr_tensor.item()} / {local_count_tensor.item()}")

def evaluate_stop_action_attack(end_token_id, dataloader, vla, processor, num_ref, args, rank, logger):
    if rank == 0:
        logger.info(f"Stop Action Attack ({num_ref} refs) adversarial prediction:")
    stop_action_asr = 0
    local_count = 0
    for (texts, images, ref_prompts) in dataloader:
        if isinstance(images[0], list):
            images = [img for sublist in images for img in sublist]

        mp_images = []
        for i in range(len(texts)):
            prompt_list = random.sample(ref_prompts[i], num_ref)
            adv_image = stop_action_token_attack(end_token_id, vla, processor, images[i], prompt_list, args.text_step, args.eps, args.alpha, args.image_step, random_start=True)
            mp_images.append(adv_image)
        mp_images = torch.cat(mp_images, dim=0)

        inputs_test = processor(images=images, text=texts, return_tensors="pt", padding=True)
        inputs_test["pixel_values"] = mp_images

        eos_count = predict_action_batch(end_token_id, vla.module, inputs_test.to(vla.module.device))

        stop_action_asr += eos_count
        local_count += len(texts)

    local_asr_tensor = torch.tensor(stop_action_asr, device='cuda')
    local_count_tensor = torch.tensor(local_count, device='cuda')
    dist.all_reduce(local_asr_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_count_tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        logger.info(f"StopAction Ours ({num_ref} refs) ASR: {local_asr_tensor.item()} / {local_count_tensor.item()}")

def evaluate_pgd(end_token_id, dataloader, vla, processor, args, rank, logger):
    if rank == 0:
        logger.info("PGD Prediction:")
    pgd_asr = 0
    local_count = 0
    text_ref_path = "path/to/dataset/bridge_orig_val_texts.txt"
    with open(text_ref_path, "r", encoding="utf-8") as f:
        text_ref = [line.strip() for line in f]

    for (texts, images, _) in dataloader:
        if isinstance(images[0], list):
            images = [img for sublist in images for img in sublist]

        inputs = processor(images=images, text=texts, return_tensors="pt", padding=True)

        prompt_list = random.sample(text_ref, len(images))
        adv_images = one_prompt_attack(end_token_id, vla, processor, images, prompt_list, epsilon=args.eps, alpha=args.alpha, step=args.image_step, random_start=True)
        inputs["pixel_values"] = adv_images

        eos_count = predict_action_batch(end_token_id, vla.module, inputs.to(vla.module.device))

        pgd_asr += eos_count
        local_count += len(texts)

    local_asr_tensor = torch.tensor(pgd_asr, device='cuda')
    local_count_tensor = torch.tensor(local_count, device='cuda')
    dist.all_reduce(local_asr_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_count_tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        logger.info(f"PGD: {local_asr_tensor.item()} / {local_count_tensor.item()}")

def evaluate_noise(end_token_id, dataloader, vla, processor, args, rank, logger):
    if rank == 0:
        logger.info("Noise Prediction:")
    noise_asr = 0
    local_count = 0
    for (texts, images, _) in dataloader:

        # List[List[Image]] -> List[Image]
        if isinstance(images[0], list):
            images = [img for sublist in images for img in sublist]

        inputs = processor(images=images, text=texts, return_tensors="pt", padding=True)
       
        # add noise to images
        ori_images = inputs["pixel_values"].clone().detach()
        inputs["pixel_values"] = add_noise_to_prismatic(ori_images, args.eps)
        inputs = inputs.to(vla.module.device, dtype=torch.bfloat16)

        eos_count = predict_action_batch(end_token_id, vla.module, inputs.to(vla.module.device))

        noise_asr += eos_count
        local_count += len(texts)

    local_asr_tensor = torch.tensor(noise_asr, device='cuda')
    local_count_tensor = torch.tensor(local_count, device='cuda')
    dist.all_reduce(local_asr_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_count_tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        logger.info(f"Noise: {local_asr_tensor.item()} / {local_count_tensor.item()}")

def main_worker(args, rank, local_rank, world_size):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)
    seed_everything(args.seed)

    model_name_or_path = "path/to/VLA/model/pi-0-cotraining-bridge"
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    pi_model = AutoModelForVision2Seq.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    ).eval().cuda(local_rank)
    vla = torch.nn.parallel.DistributedDataParallel(pi_model, device_ids=[local_rank])

    dataset_path = f"path/to/VLA/dataset/gpt/{args.dataset}/{args.dataset}_sampled.pt"
    dataset = torch.load(dataset_path, weights_only=False)
    
    # Sample only on global rank 0, then broadcast to all processes
    if rank == 0:
        dataset_attack = random.sample(dataset, args.max_samples)
    else:
        dataset_attack = None
    dataset_attack_list = [dataset_attack]
    dist.broadcast_object_list(dataset_attack_list, src=0)
    dataset_attack = dataset_attack_list[0]

    text_ref_path = "path/to/VLA/dataset/libero_spatial_no_noops_texts.txt"

    with open(text_ref_path, "r", encoding="utf-8") as f:
        # text_ref = [line.strip() for line in f]
        text_ref = [
            line.strip().replace("What action should the robot take to ", "").replace("?", ".")
            for line in f
        ]
    text_ref_20 = random.sample(text_ref, 20)
    text_ref_15 = random.sample(text_ref_20, 15)
    text_ref_10 = random.sample(text_ref_15, 10)
    text_ref_5  = random.sample(text_ref_10, 5)
    text_ref_1  = random.sample(text_ref_5, 1)
    # if local_rank == 0:
    #     print(text_ref_20)

    logger = get_logger(args)


    sampler = DistributedSampler(PackedDatasetWithRef(dataset_attack), num_replicas=world_size, rank=rank, shuffle=True)
    logger.info(f"rank{rank} indices: {list(sampler)}")
    dataloader = DataLoader(
        PackedDatasetWithRef(dataset_attack), 
        batch_size=16, 
        shuffle=False,  # Set shuffle=False when using sampler
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=custom_collate_fn_with_ref,
        sampler=sampler
    )

    processor.tokenizer.padding_side = 'left'
    end_token_id = processor.tokenizer.eos_token_id
    # Output all token ranges

    if args.attack == "baseline_ref":
        evaluate_multi_prompt_ref(end_token_id, dataloader, vla, processor, text_ref_20, args, rank, logger)
        evaluate_multi_prompt_ref(end_token_id, dataloader, vla, processor, text_ref_15, args, rank, logger)
        evaluate_multi_prompt_ref(end_token_id, dataloader, vla, processor, text_ref_10, args, rank, logger)
        evaluate_multi_prompt_ref(end_token_id, dataloader, vla, processor, text_ref_5, args, rank, logger)

    elif args.attack == "baseline_gpt":
        evaluate_multi_prompt_attack(end_token_id, dataloader, vla, processor, 20, args, rank, logger)
        evaluate_multi_prompt_attack(end_token_id, dataloader, vla, processor, 15, args, rank, logger)
        evaluate_multi_prompt_attack(end_token_id, dataloader, vla, processor, 10, args, rank, logger)
        evaluate_multi_prompt_attack(end_token_id, dataloader, vla, processor, 5, args, rank, logger)

    elif args.attack == "ours_ref":
        evaluate_stop_action_ref(end_token_id, dataloader, vla, processor, text_ref_20, args, rank, logger)
        evaluate_stop_action_ref(end_token_id, dataloader, vla, processor, text_ref_15, args, rank, logger)
        evaluate_stop_action_ref(end_token_id, dataloader, vla, processor, text_ref_10, args, rank, logger)
        evaluate_stop_action_ref(end_token_id, dataloader, vla, processor, text_ref_5, args, rank, logger)

    elif args.attack == "ours_gpt":
        evaluate_stop_action_attack(end_token_id, dataloader, vla, processor, 20, args, rank, logger)
        # evaluate_stop_action_attack(end_token_id, dataloader, vla, processor, 15, args, rank, logger)
        # evaluate_stop_action_attack(end_token_id, dataloader, vla, processor, 10, args, rank, logger)
        # evaluate_stop_action_attack(end_token_id, dataloader, vla, processor, 5, args, rank, logger)

    elif args.attack == "noise":
        evaluate_noise(end_token_id, dataloader, vla, processor, args, rank, logger)

    elif args.attack == "pgd":
        evaluate_pgd(end_token_id, dataloader, vla, processor, args, rank, logger)

    else:
        raise ValueError(f"Invalid attack type: {args.attack}")


    if rank == 0:
        logger.info(args)
        logger.info(dataset_path)
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--text_step", type=int, default=10)
    parser.add_argument("--image_step", type=int, default=100)
    parser.add_argument("--eps", type=int, default=4)
    parser.add_argument("--alpha", type=int, default=1)
    parser.add_argument("--attack", type=str, default="ours")
    parser.add_argument("--dataset", type=str, default="libero_10_no_noops")
    args = parser.parse_args()

    args.eps = args.eps / 255.0
    args.alpha = args.alpha / 255.0

    # Get global rank, local rank, and world_size
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ['WORLD_SIZE'])
    
    # torchrun --nproc_per_node=8 run_pi_gpt_ddp.py --max_samples 256 --text_step 20 --attack baseline_ref
    # torchrun --nproc_per_node=8 --nnodes=2 --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_pi_gpt_ddp.py --max_samples 256 --text_step 20 --attack baseline_ref
    main_worker(args, rank, local_rank, world_size)