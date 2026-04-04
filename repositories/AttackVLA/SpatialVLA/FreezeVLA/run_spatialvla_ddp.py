import os
import torch
import random
import argparse
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModel, AutoProcessor
from utils_spatialvla import *

def evaluate_multi_prompt_ref(dataloader, vla, processor, ref, args, rank):
    if rank == 0:
        logger.info(f"Multi Prompt Attack ({len(ref)} refs):")
    multi_prompt_asr = 0
    local_count = 0
    for (texts, images, _) in dataloader:
        mp_images = []
        for i in range(len(texts)):

            adv_image = multi_prompt_attack(vla, processor, images[i], ref, args.eps, args.alpha, args.image_step, random_start=True)
            mp_images.append(adv_image)
        mp_images = torch.stack(mp_images, dim=0)

        inputs_test = processor(images=images, text=texts, return_tensors="pt", padding=True)
        inputs_test["pixel_values"] = mp_images
        generation_outputs_test = vla.module.predict_action(inputs_test)
        multi_prompt_asr += calculate_action(generation_outputs_test, processor)
        local_count += len(texts)

    local_asr_tensor = torch.tensor(multi_prompt_asr, device='cuda')
    local_count_tensor = torch.tensor(local_count, device='cuda')
    dist.all_reduce(local_asr_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_count_tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        logger.info(f"Multi Prompt Attack ({len(ref)} refs) ASR: {local_asr_tensor.item()} / {local_count_tensor.item()}")

def evaluate_multi_prompt_attack(dataloader, vla, processor, num_ref, args, rank):
    if rank == 0:
        logger.info(f"Multi Prompt Attack ({num_ref} refs):")
    multi_prompt_asr = 0
    local_count = 0
    for (texts, images, ref_prompts) in dataloader:
        mp_images = []
        for i in range(len(texts)):
            prompt_list = random.sample(ref_prompts[i], num_ref)
            adv_image = multi_prompt_attack(vla, processor, images[i], prompt_list, args.eps, args.alpha, args.image_step, random_start=True)
            mp_images.append(adv_image)
        mp_images = torch.stack(mp_images, dim=0)

        inputs_test = processor(images=images, text=texts, return_tensors="pt", padding=True)
        inputs_test["pixel_values"] = mp_images
        generation_outputs_test = vla.module.predict_action(inputs_test)
        multi_prompt_asr += calculate_action(generation_outputs_test, processor)
        local_count += len(texts)

    local_asr_tensor = torch.tensor(multi_prompt_asr, device='cuda')
    local_count_tensor = torch.tensor(local_count, device='cuda')
    dist.all_reduce(local_asr_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_count_tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        logger.info(f"Multi Prompt Attack ({num_ref} refs) ASR: {local_asr_tensor.item()} / {local_count_tensor.item()}")

def evaluate_stop_action_ref(dataloader, vla, processor, ref, args, rank):
    if rank == 0:
        logger.info(f"Stop Action Attack ({len(ref)} refs):")
    stop_action_asr = 0
    local_count = 0
    for (texts, images, _) in dataloader:
        mp_images = []
        for i in range(len(texts)):

            adv_image = stop_action_token_attack(vla, processor, images[i], ref, args.text_step, args.eps, args.alpha, args.image_step, random_start=True)
            mp_images.append(adv_image)
        mp_images = torch.stack(mp_images, dim=0)

        inputs_test = processor(images=images, text=texts, return_tensors="pt", padding=True)
        inputs_test["pixel_values"] = mp_images
        generation_outputs_test = vla.module.predict_action(inputs_test)
        stop_action_asr += calculate_action(generation_outputs_test, processor)
        local_count += len(texts)

    local_asr_tensor = torch.tensor(stop_action_asr, device='cuda')
    local_count_tensor = torch.tensor(local_count, device='cuda')
    dist.all_reduce(local_asr_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_count_tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        logger.info(f"StopAction Ours ({len(ref)} refs) ASR: {local_asr_tensor.item()} / {local_count_tensor.item()}")

def evaluate_stop_action_attack(dataloader, vla, processor, num_ref, args, rank):
    if rank == 0:
        logger.info(f"Stop Action Attack ({num_ref} refs):")
    stop_action_asr = 0
    local_count = 0
    for (texts, images, ref_prompts) in dataloader:
        mp_images = []
        for i in range(len(texts)):
            prompt_list = random.sample(ref_prompts[i], num_ref)
            adv_image = stop_action_token_attack(vla, processor, images[i], prompt_list, args.text_step, args.eps, args.alpha, args.image_step, random_start=True)
            mp_images.append(adv_image)
        mp_images = torch.stack(mp_images, dim=0)

        inputs_test = processor(images=images, text=texts, return_tensors="pt", padding=True)
        inputs_test["pixel_values"] = mp_images
        generation_outputs_test = vla.module.predict_action(inputs_test)
        stop_action_asr += calculate_action(generation_outputs_test, processor)
        local_count += len(texts)

    local_asr_tensor = torch.tensor(stop_action_asr, device='cuda')
    local_count_tensor = torch.tensor(local_count, device='cuda')
    dist.all_reduce(local_asr_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_count_tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        logger.info(f"StopAction Ours ({num_ref} refs) ASR: {local_asr_tensor.item()} / {local_count_tensor.item()}")

def evaluate_pgd(dataloader, vla, processor, args, rank):
    if rank == 0:
        logger.info("PGD Prediction:")
    pgd_asr = 0
    local_count = 0
    text_ref_path = "/path/to/model/folder/"
    with open(text_ref_path, "r", encoding="utf-8") as f:
        text_ref = [line.strip() for line in f]

    for (texts, images, _) in dataloader:
        inputs = processor(images=images, text=texts, return_tensors="pt", padding=True)

        prompt_list = random.sample(text_ref, len(images))
        _, adv_images = one_prompt_attack(vla, processor, images, prompt_list, epsilon=args.eps, alpha=args.alpha, step=args.image_step, random_start=True)
        inputs["pixel_values"] = adv_images
        pgd_outputs = vla.module.predict_action(inputs)
        pgd_asr += calculate_action(pgd_outputs, processor)
        local_count += len(texts)

    local_asr_tensor = torch.tensor(pgd_asr, device='cuda')
    local_count_tensor = torch.tensor(local_count, device='cuda')
    dist.all_reduce(local_asr_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_count_tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        logger.info(f"PGD: {local_asr_tensor.item()} / {local_count_tensor.item()}")

def evaluate_noise(dataloader, vla, processor, args, rank):
    if rank == 0:
        logger.info("Noise Prediction:")
    noise_asr = 0
    local_count = 0
    for (texts, images, _) in dataloader:
        inputs = processor(images=images, text=texts, return_tensors="pt", padding=True)
       
        # add noise to images
        ori_images = inputs["pixel_values"].clone().detach()
        noise_images = ori_images + torch.empty_like(ori_images).uniform_(-args.eps, args.eps)
        
        noise_images = torch.clamp(noise_images, min=0, max=1).detach()
        inputs["pixel_values"] = noise_images

        noise_outputs = vla.module.predict_action(inputs)
        noise_asr += calculate_action(noise_outputs, processor)
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

    model_name_or_path = "/path/to/model/folder/SpatialVLA"
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    vla = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16).eval().cuda(local_rank)
    vla = torch.nn.parallel.DistributedDataParallel(vla, device_ids=[local_rank])

    dataset_path = f"/path/to/dataset/folder/gpt/{args.dataset}/{args.dataset}_sampled.pt"
    dataset = torch.load(dataset_path, weights_only=False)
    
    if rank == 0:
        dataset_attack = random.sample(dataset, args.max_samples)
    else:
        dataset_attack = None
    dataset_attack_list = [dataset_attack]
    dist.broadcast_object_list(dataset_attack_list, src=0)
    dataset_attack = dataset_attack_list[0]

    text_ref_path = "/path/to/dataset/folder/libero_goal_no_noops_texts.txt"

    with open(text_ref_path, "r", encoding="utf-8") as f:
        text_ref = [line.strip() for line in f]


    sampler = DistributedSampler(PackedDatasetWithRef(dataset_attack), num_replicas=world_size, rank=rank, shuffle=True)
    logger.info(f"rank{rank} indices: {list(sampler)}")
    dataloader = DataLoader(
        PackedDatasetWithRef(dataset_attack), 
        batch_size=16, 
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=custom_collate_fn_with_ref,
        sampler=sampler
    )


    if args.attack == "multi_ref":
        evaluate_multi_prompt_ref(dataloader, vla, processor, random.sample(text_ref, 20), args, rank)

    elif args.attack == "multi_gpt":
        evaluate_multi_prompt_attack(dataloader, vla, processor, 20, args, rank)

    elif args.attack == "ours_ref":
        evaluate_stop_action_ref(dataloader, vla, processor, random.sample(text_ref, 20), args, rank)

    elif args.attack == "ours_gpt":
        evaluate_stop_action_attack(dataloader, vla, processor, 20, args, rank)

    elif args.attack == "noise":
        evaluate_noise(dataloader, vla, processor, args, rank)

    elif args.attack == "pgd":
        evaluate_pgd(dataloader, vla, processor, args, rank)

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

    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ['WORLD_SIZE'])

    logger = get_logger(args)

    # torchrun --nproc_per_node=8 run_spatialvla_gpt_ddp.py --max_samples 256 --attack ours_gpt
    # torchrun --nproc_per_node=8 --nnodes=4 --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_spatialvla_gpt_ddp.py --max_samples 256 --attack ours_gpt
    main_worker(args, rank, local_rank, world_size)