import os
import copy
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import string
from nltk.corpus import wordnet
import re
import gc
import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler

class PackedDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        prompt = item['text']
        images = item['images']
        return prompt, images

class PackedDatasetWithRef(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        prompt = item['prompt']
        images = item['image']
        ref_prompts = item['ref_prompts']
        return prompt, images, ref_prompts

# def custom_collate_fn(batch):
#     prompts = [item[0] for item in batch]
#     images = [item[1] for item in batch]
#     # 预处理文本，添加前缀
#     formatted_prompts = [f"In: What action should the robot take to {prompt}?\nOut:" for prompt in prompts]
#     return formatted_prompts, images

def custom_collate_fn_with_ref(batch):
    prompts = [item[0] for item in batch]
    images = [item[1] for item in batch]
    ref_prompts = [item[2] for item in batch]
    formatted_prompts = [f"In: What action should the robot take to {prompt}?\nOut:" for prompt in prompts]
    formatted_ref_prompts = [[f"In: {p}\nOut:" for p in prompt_list] for prompt_list in ref_prompts]
    return formatted_prompts, images, formatted_ref_prompts

def calculate_action(generation_outputs, processor):
    count = 0
    target_array = np.array([[1,0,0],[0,0,0],[0,0,0],[0,0,0]])

    for generation_output in generation_outputs:
        single_action = processor.decode_actions(generation_output.unsqueeze(0), unnorm_key="bridge_orig/1.0.0")
        action_ids_value = single_action.get("action_ids")
        current_action_ids_array = np.array(action_ids_value)

        # print(current_action_ids_array)
        if np.array_equal(current_action_ids_array, target_array):
            count += 1
    
    return count

def get_logger(args):
    log_filename = f"log/{args.dataset}/run_openvla_{args.attack}_{args.dataset}_{int(args.eps * 255)}.log"
    logger = logging.getLogger("openvla_logger")  # 固定logger名字，防止多进程/多线程混乱
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 防止日志重复

    # 只添加一次handler，防止重复
    if not logger.handlers:
        file_handler = ConcurrentRotatingFileHandler(log_filename, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(stream_handler)

    return logger

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def one_prompt_attack(end_token_id, model, processor, images, prompts, epsilon=4/255, alpha=1/255, step=100, random_start=True):
    """
    Performs a PGD adversarial attack on the input image to maximize the probability of the EOS (End Of Sequence) token for the generated token.
    The attack is designed to MINIMIZE the CrossEntropyLoss for the EOS token, thereby maximizing its probability in the model's output.

    Args:
        model: The VLA model to be attacked.
        processor: The processor used for preprocessing images and text for the model.
        image: The original input PIL image.
        prompt: The text prompt to be used for generation.
        epsilon: Maximum allowed L-infinity norm perturbation (default: 4/255).
        alpha: Step size for each PGD iteration (default: 1/255).
        step: Number of PGD iterations to perform (default: 100).
        random_start: Whether to start from a random point within the epsilon ball (default: True).

    Returns:
        adv_inputs: The adversarially perturbed inputs, ready to be fed into the VLA model.
    """
    model.eval()
    device = model.device

    # print(model.vocab_size, eos_token_id)

    ori_inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True).to(torch.bfloat16).to(device)
    ori_images = inverse_norm(ori_inputs["pixel_values"].clone().detach())

    adv_images = ori_images.clone().detach().requires_grad_(True)

    # Starting at a uniformly random point
    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    input_ids = ori_inputs["input_ids"]
    attention_mask = ori_inputs["attention_mask"]
    # if input_ids is not None:
    #     need_pad = input_ids[:, -1] != 29871  # (batch,)
    #     if need_pad.any():
    #         pad = torch.full((input_ids.size(0), 1), 29871, dtype=input_ids.dtype, device=input_ids.device)
    #         input_ids = torch.cat([input_ids, pad], dim=1)
            
    #         mask_pad = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)
    #         attention_mask = torch.cat([attention_mask, mask_pad], dim=1)
            
    #         input_ids[~need_pad, -1] = input_ids[~need_pad, -2]

    for _ in range(step):
        adv_images.requires_grad = True
        adv_pixel_values = set_norm(adv_images).to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=adv_pixel_values,
            return_dict=True
        )

        # Calculate loss
        logits = outputs.logits
        shift_logits = logits[:, -1 :, :]
        
        probs = F.log_softmax(shift_logits, dim=-1)
        eos_probs = probs[..., end_token_id]
        loss = torch.sum(eos_probs)

        # Print the max index of probs
        # print("Max Index:", torch.argmax(probs, dim=-1), "Loss:", loss)

        # Update adversarial images
        grad       = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
        adv_images = adv_images.detach() + alpha * grad.sign()
        delta      = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(ori_images + delta, min=0, max=1).detach()
        
        torch.cuda.empty_cache()

    return set_norm(adv_images).to(device)

def multi_prompt_attack(end_token_id, model, processor, image, prompt_list, epsilon=4/255, alpha=1/255, step=100, random_start=True):
    """
    针对同一张图片和多个prompt，实现跨prompt的对抗攻击。
    优化目标是在所有prompt下的输出都被扰动。
    """
    model.eval()
    device = model.device
    model = model.module if hasattr(model, 'module') else model

    with torch.no_grad():
        ori_inputs = processor(images=[image]*len(prompt_list), text=prompt_list, return_tensors="pt", padding=True).to(torch.bfloat16).to(device)
        ori_images = inverse_norm(ori_inputs["pixel_values"][0].unsqueeze(0).clone().detach())

    adv_images = ori_images.clone().detach()

    if random_start:
        # Starting at a uniformly random point
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(step):
        adv_images.requires_grad = True
        adv_pixel_values = set_norm(adv_images).to(device)

        # Calculate loss
        outputs = model(
            input_ids=ori_inputs["input_ids"],
            attention_mask=ori_inputs["attention_mask"],
            pixel_values=adv_pixel_values.repeat(len(prompt_list), 1, 1, 1),
            return_dict=True
        )
        logits = outputs.logits
        shift_logits = logits[:, -1 :, :]
        probs = F.log_softmax(shift_logits, dim=-1)
        eos_probs = probs[..., end_token_id]
        loss = torch.sum(eos_probs)

        # Print the max index of probs
        # print("Max Index:", torch.argmax(probs, dim=-1), "Loss:", loss)

        # if torch.sum(torch.argmax(probs, dim=-1)) == len(prompt_list):
        #     break

        # Update adversarial images
        grad       = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
        adv_images = adv_images.detach() + alpha * grad.sign()
        delta      = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(ori_images + delta, min=0, max=1).detach()

        del outputs, grad
        torch.cuda.empty_cache()

    return set_norm(adv_images).to(device)

def stop_action_token_attack(end_token_id, model, processor, image, prompt_list, text_step=10, epsilon=4/255, alpha=1/255, image_step=100, random_start=True):

    model.eval()
    device = model.device
    model = model.module if hasattr(model, 'module') else model

    with torch.no_grad():
        ori_inputs = processor(images=[image]*len(prompt_list), text=prompt_list, return_tensors="pt", padding=True).to(torch.bfloat16).to(device)
        ori_images = inverse_norm(ori_inputs["pixel_values"][0].unsqueeze(0).clone().detach())
    
    adv_images = ori_images.clone().detach()

    llm = model.language_model.get_input_embeddings()
    ori_embeds = llm(ori_inputs["input_ids"]).detach()
    adv_embeds = ori_embeds.clone().detach()
    mask = make_mask(ori_embeds, ori_inputs, model, processor)

    if random_start:
        # Starting at a uniformly random point
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for i in range(image_step):
        # optimize adversarial text embedding, find the hard prompt
        if i % 50 == 0:
            for _ in range(text_step):
                adv_embeds.requires_grad = True

                # openvl不支持inputs_embeds走完全程，如果用input_ids又会覆盖inputs_embeds，记得修改
                adv_text_inputs = {
                    "input_ids": ori_inputs['input_ids'],
                    "inputs_embeds": adv_embeds.clone(),
                    "attention_mask": ori_inputs['attention_mask'],
                    "pixel_values": set_norm(adv_images).detach().repeat(len(prompt_list), 1, 1, 1),
                }

                # Calculate loss
                outputs_text = model(**adv_text_inputs, return_dict=True)
                logits_text = outputs_text.logits
                shift_logits_text = logits_text[:, -1 :, :]
                probs_text = F.log_softmax(shift_logits_text, dim=-1)
                eos_probs_text = probs_text[..., end_token_id]

                # detach 保存，不带梯度，防止保留整个计算图
                per_prompt_loss = eos_probs_text.squeeze(1).detach()
                loss_text = torch.sum(eos_probs_text)

                # Print the max index of probs
                # print("Max Index:", torch.argmax(probs_text, dim=-1), "Text Loss:", loss_text)
                
                # Calculate gradients
                grad_text = torch.autograd.grad(loss_text, adv_embeds, retain_graph=False, create_graph=False)[0]
                grad_text_norm = torch.norm(grad_text * mask, dim=-1)   # [B, seq_len]
                max_grad_idx = torch.argmax(grad_text_norm, dim=1)      # [B]

                adv_embeds = adv_embeds.detach()
                # 清理显存（包括 outputs_text 相关大张量）
                del outputs_text, logits_text, shift_logits_text, probs_text, eos_probs_text, loss_text, grad_text, grad_text_norm
                torch.cuda.empty_cache()

                with torch.no_grad():
                    sel_token_ids = ori_inputs["input_ids"].gather(1, max_grad_idx.unsqueeze(1)).squeeze(1)
                    token_readables = processor.tokenizer.batch_decode(sel_token_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)

                    new_prompt_list = prompt_list.copy()
                    for b, token_readable in enumerate(token_readables):
                        # idx = max_grad_idx[b].item()

                        # Clean token text (remove leading/trailing whitespace)
                        token_clean = token_readable.strip()

                        # 用正则分词，保留连字符和特殊引号'
                        # prompt_words = re.findall(r"[A-Za-z]\.(?:[A-Za-z]\.)+|\w+(?:[-'']\w+)*|\?", prompt_list[b])
                        prompt_words = re.findall(r"\w+(?:[-]\w+)*|\?", prompt_list[b])

                        try:
                            word_pos = [w.lower() for w in prompt_words].index(token_clean.lower())
                            select_word = prompt_words[word_pos]
                            # print(f"Prompt[{b}] max gradient token: {select_word}  (word_pos_in_prompt={word_pos})")
                        except ValueError:
                            matches = [i for i, w in enumerate(prompt_words) if token_clean.lower() in w.lower()]
                            if matches:
                                word_pos = matches[0]
                                select_word = prompt_words[word_pos]
                                # print(f"Prompt[{b}] max gradient token substring: {token_clean} -> use full word '{select_word}'  (word_idx_in_prompt={word_pos})")
                            else:
                                raise ValueError(f"Warning: Prompt[{b}], '{prompt_list[b]}' cannot find token '{token_clean}'!")
                        
                        # 替换select_word为近义词（保留原句标点和格式）
                        synonyms = get_synonym(select_word)

                        if synonyms:
                            synonym = random.choice(synonyms)
                            # print(select_word, synonym)
                            # print("ori prompt: ", prompt_list[b])
                            pattern = re.compile(r'\b{}\b'.format(re.escape(select_word)), re.IGNORECASE)
                            new_prompt_list[b] = pattern.sub(synonym, prompt_list[b], count=1)
                            # print("new prompt: ", new_prompt_list[b])

                    # 替换prompt_list后，重新生成ori_inputs、ori_embeds和mask，保证shape一致
                    inputs_temp = processor(images=[image]*len(new_prompt_list), text=new_prompt_list, return_tensors="pt", padding=True).to(torch.bfloat16).to(device)
                    inputs_temp["pixel_values"] = set_norm(adv_images).repeat(len(new_prompt_list), 1, 1, 1)
                    outputs_temp = model(**inputs_temp, return_dict=True)
                    logits_text_temp = outputs_temp.logits
                    shift_logits_text_temp = logits_text_temp[:, -1 :, :]
                    probs_text_temp = F.log_softmax(shift_logits_text_temp, dim=-1)
                    eos_probs_text_temp = probs_text_temp[..., end_token_id]
                    per_prompt_loss_temp = eos_probs_text_temp.squeeze(1)
                    
                    # 比较并更新prompt_list
                    for j in range(len(prompt_list)):
                        if per_prompt_loss_temp[j] <= per_prompt_loss[j]:
                            prompt_list[j] = new_prompt_list[j]
                            # print(f"New Loss: {per_prompt_loss_temp[j]}, Ori Loss: {per_prompt_loss[j]}")
                            # print(f"Prompt[{j}] changed from {prompt_list[j]} to {new_prompt_list[j]}")

                    # 用最新的prompt_list生成ori_inputs、ori_embeds、mask、adv_embeds
                    ori_inputs = processor(images=[image]*len(prompt_list), text=prompt_list, return_tensors="pt", padding=True).to(torch.bfloat16).to(device)
                    ori_embeds = llm(ori_inputs["input_ids"]).detach()
                    adv_embeds = ori_embeds.clone().detach()  # 保证adv_embeds形状同步
                    mask = make_mask(ori_embeds, ori_inputs, model, processor)

                    # 用完临时变量后立即del，节省显存
                    del inputs_temp, outputs_temp, logits_text_temp, shift_logits_text_temp, probs_text_temp, eos_probs_text_temp, per_prompt_loss_temp
                    torch.cuda.empty_cache()
                    gc.collect()

            torch.cuda.empty_cache()
            gc.collect()

        torch.cuda.empty_cache()
        gc.collect()
        adv_images.requires_grad = True
        adv_image_inputs = {
            "input_ids": ori_inputs['input_ids'],
            "attention_mask": ori_inputs['attention_mask'],
            "pixel_values": set_norm(adv_images).repeat(len(prompt_list), 1, 1, 1),
        }

        # Calculate loss
        outputs_image = model(**adv_image_inputs, return_dict=True)
        logits_image = outputs_image.logits
        shift_logits_image = logits_image[:, -1 :, :]
        probs_image = F.log_softmax(shift_logits_image, dim=-1)
        eos_probs_image = probs_image[..., end_token_id]
        loss_image = torch.sum(eos_probs_image)

        # Print the max index of probs
        # print("Max Index:", torch.argmax(probs_image, dim=-1), "Image Loss:", loss_image)

        # Calculate gradients
        grad_img = torch.autograd.grad(loss_image, adv_images, retain_graph=False, create_graph=False)[0]

        # Update adversarial images
        adv_images = adv_images.detach() + alpha * grad_img.sign()
        delta_img = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(ori_images + delta_img, min=0, max=1).detach()

        # 显存释放
        del outputs_image, logits_image, shift_logits_image, probs_image, eos_probs_image, loss_image, grad_img, adv_image_inputs
        torch.cuda.empty_cache()
        gc.collect()
    
    return set_norm(adv_images).to(device)

def stop_action_token_batch_attack(end_token_id, model, processor, image, prompt_list, text_step=10, epsilon=4/255, alpha=1/255, image_step=100, random_start=True):

    model.eval()
    device = model.device
    model = model.module if hasattr(model, 'module') else model

    eos_token_id = end_token_id

    with torch.no_grad():
        ori_inputs = processor(images=[image]*len(prompt_list), text=prompt_list, return_tensors="pt", padding=True).to(torch.bfloat16).to(device)
        ori_images = inverse_norm(ori_inputs["pixel_values"][0].unsqueeze(0).clone().detach())
    
    adv_images = ori_images.clone().detach()

    llm = model.language_model.get_input_embeddings()
    ori_embeds = llm(ori_inputs["input_ids"]).detach()
    adv_embeds = ori_embeds.clone().detach()
    mask = make_mask(ori_embeds, ori_inputs, model, processor)

    batch_size = 10

    if random_start:
        # Starting at a uniformly random point
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for i in range(image_step):
        # optimize adversarial text embedding, find the hard prompt
        if i % 50 == 0:
            for _ in range(text_step):
                adv_embeds.requires_grad = True

                num_prompts = len(prompt_list)
                num_batches = (num_prompts + batch_size - 1) // batch_size

                # 改为"分批前向-立即反向-累积梯度"
                adv_embeds_grad = torch.zeros_like(adv_embeds)  # 累积各 batch 的梯度
                all_per_prompt_loss = []

                for batch_idx in range(num_batches):
                    start = batch_idx * batch_size
                    end = min((batch_idx + 1) * batch_size, num_prompts)

                    batch_input_ids = ori_inputs['input_ids'][start:end]
                    batch_attention_mask = ori_inputs['attention_mask'][start:end]

                    # 创建当前 batch 的可求梯度副本
                    batch_embeds = adv_embeds[start:end].clone().requires_grad_(True)
                    batch_pixel_values = set_norm(adv_images).repeat(end - start, 1, 1, 1)

                    adv_text_inputs_batch = {
                        "input_ids": batch_input_ids,
                        "inputs_embeds": batch_embeds,
                        "attention_mask": batch_attention_mask,
                        "pixel_values": batch_pixel_values,
                    }

                    # Calculate loss
                    outputs_text = model(**adv_text_inputs_batch, return_dict=True)
                    logits_text = outputs_text.logits
                    shift_logits_text = logits_text[:, -1 :, :]
                    probs_text = F.log_softmax(shift_logits_text, dim=-1)
                    eos_probs_text = probs_text[..., eos_token_id]

                    # 保存无梯度 per_prompt_loss，用于后续替换 prompt
                    per_prompt_loss_batch = eos_probs_text.squeeze(1)
                    all_per_prompt_loss.append(per_prompt_loss_batch.detach())

                    # 计算当前 batch 的梯度并累积
                    loss_batch = eos_probs_text.sum()
                    grad_batch = torch.autograd.grad(loss_batch, batch_embeds, retain_graph=False, create_graph=False)[0]
                    adv_embeds_grad[start:end] = grad_batch.detach()

                    # 显存释放
                    del outputs_text, logits_text, shift_logits_text, probs_text, eos_probs_text, per_prompt_loss_batch, loss_batch, grad_batch
                    torch.cuda.empty_cache()
                    gc.collect()

                # 拼回完整的 per_prompt_loss
                per_prompt_loss = torch.cat(all_per_prompt_loss, dim=0)

                # Print the max index of probs
                # print("Max Index:", torch.argmax(probs_text, dim=-1), "Text Loss:", loss_text)
                
                # Calculate gradients
                grad_text = adv_embeds_grad

                grad_text_norm = torch.norm(grad_text * mask, dim=-1)   # [B, seq_len]
                max_grad_idx = torch.argmax(grad_text_norm, dim=1)      # [B]

                with torch.no_grad():
                    sel_token_ids = ori_inputs["input_ids"].gather(1, max_grad_idx.unsqueeze(1)).squeeze(1)
                    token_readables = processor.tokenizer.batch_decode(sel_token_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)

                    new_prompt_list = prompt_list.copy()
                    for b, token_readable in enumerate(token_readables):
                        # idx = max_grad_idx[b].item()

                        # Clean token text (remove leading/trailing whitespace)
                        token_clean = token_readable.strip()

                        # 用正则分词，保留连字符和特殊引号'
                        # prompt_words = re.findall(r"[A-Za-z]\.(?:[A-Za-z]\.)+|\w+(?:[-'']\w+)*|\?", prompt_list[b])
                        prompt_words = re.findall(r"\w+(?:[-]\w+)*|\?", prompt_list[b])

                        try:
                            word_pos = [w.lower() for w in prompt_words].index(token_clean.lower())
                            select_word = prompt_words[word_pos]
                            # print(f"Prompt[{b}] max gradient token: {select_word}  (word_pos_in_prompt={word_pos})")
                        except ValueError:
                            matches = [i for i, w in enumerate(prompt_words) if token_clean.lower() in w.lower()]
                            if matches:
                                word_pos = matches[0]
                                select_word = prompt_words[word_pos]
                                # print(f"Prompt[{b}] max gradient token substring: {token_clean} -> use full word '{select_word}'  (word_idx_in_prompt={word_pos})")
                            else:
                                raise ValueError(f"Warning: Prompt[{b}], '{prompt_list[b]}' cannot find token '{token_clean}'!")
                        
                        # 替换select_word为近义词（保留原句标点和格式）
                        synonyms = get_synonym(select_word)

                        if synonyms:
                            synonym = random.choice(synonyms)
                            # print(select_word, synonym)
                            # print("ori prompt: ", prompt_list[b])
                            pattern = re.compile(r'\b{}\b'.format(re.escape(select_word)), re.IGNORECASE)
                            new_prompt_list[b] = pattern.sub(synonym, prompt_list[b], count=1)
                            # print("new prompt: ", new_prompt_list[b])

                    # 替换prompt_list后，重新生成ori_inputs、ori_embeds和mask，保证shape一致
                    inputs_temp = processor(images=[image]*len(new_prompt_list), text=new_prompt_list, return_tensors="pt", padding=True).to(torch.bfloat16).to(device)
                    inputs_temp["pixel_values"] = set_norm(adv_images).repeat(len(new_prompt_list), 1, 1, 1)
                    outputs_temp = model(**inputs_temp, return_dict=True)
                    logits_text_temp = outputs_temp.logits
                    shift_logits_text_temp = logits_text_temp[:, -1 :, :]
                    probs_text_temp = F.log_softmax(shift_logits_text_temp, dim=-1)
                    eos_probs_text_temp = probs_text_temp[..., eos_token_id]
                    per_prompt_loss_temp = eos_probs_text_temp.squeeze(1)
                    
                    # 比较并更新prompt_list
                    for j in range(len(prompt_list)):
                        if per_prompt_loss_temp[j] <= per_prompt_loss[j]:
                            prompt_list[j] = new_prompt_list[j]
                            # print(f"New Loss: {per_prompt_loss_temp[j]}, Ori Loss: {per_prompt_loss[j]}")
                            # print(f"Prompt[{j}] changed from {prompt_list[j]} to {new_prompt_list[j]}")

                    # 用最新的prompt_list生成ori_inputs、ori_embeds、mask、adv_embeds
                    ori_inputs = processor(images=[image]*len(prompt_list), text=prompt_list, return_tensors="pt", padding=True).to(torch.bfloat16).to(device)
                    ori_embeds = llm(ori_inputs["input_ids"]).detach()
                    adv_embeds = ori_embeds.clone().detach()  # 保证adv_embeds形状同步
                    mask = make_mask(ori_embeds, ori_inputs, model, processor)

                    # 用完临时变量后立即del，节省显存
                    del inputs_temp, outputs_temp, logits_text_temp, shift_logits_text_temp, probs_text_temp, eos_probs_text_temp, per_prompt_loss_temp
                    torch.cuda.empty_cache()
                    gc.collect()
        
        # ---------------- Image PGD：分批前向-立即反向-累积梯度 ---------------- #
        adv_images.requires_grad = True

        grad_img_total = torch.zeros_like(adv_images)
        num_prompts_img = len(prompt_list)
        num_batches_img = (num_prompts_img + batch_size - 1) // batch_size

        for b_idx in range(num_batches_img):
            s = b_idx * batch_size
            e = min((b_idx + 1) * batch_size, num_prompts_img)

            batch_input_ids = ori_inputs['input_ids'][s:e]
            batch_attention_mask = ori_inputs['attention_mask'][s:e]
            batch_pixel_values = set_norm(adv_images).repeat(e - s, 1, 1, 1)

            adv_image_inputs_batch = {
                "input_ids": batch_input_ids,
                "attention_mask": batch_attention_mask,
                "pixel_values": batch_pixel_values,
            }

            # Calculate loss
            outputs_img = model(**adv_image_inputs_batch, return_dict=True)
            logits_img = outputs_img.logits
            shift_logits_img = logits_img[:, -1 :, :]
            probs_img = F.log_softmax(shift_logits_img, dim=-1)
            eos_probs_img = probs_img[..., eos_token_id]

            # Print the max index of probs
            # print("Max Index:", torch.argmax(probs_image, dim=-1), "Image Loss:", loss_image)

            # Calculate gradients
            loss_img_batch = eos_probs_img.sum()
            grad_img_batch = torch.autograd.grad(loss_img_batch, adv_images, retain_graph=False, create_graph=False)[0]
            grad_img_total += grad_img_batch.detach()

            # 显存释放
            del outputs_img, logits_img, shift_logits_img, probs_img, eos_probs_img, loss_img_batch, grad_img_batch
            torch.cuda.empty_cache()
            gc.collect()

        # Update adversarial images
        adv_images = adv_images.detach() + alpha * grad_img_total.sign()
        delta_img = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(ori_images + delta_img, min=0, max=1).detach()

    torch.cuda.empty_cache()
    
    return set_norm(adv_images).to(device)

def make_mask(ori_embeds, ori_inputs, model, processor):
    pad_token_id = processor.tokenizer.pad_token_id
    pad_counts = (ori_inputs["input_ids"] == pad_token_id).sum(dim=1)
    prefix = pad_counts + 10 # 10 is the length of "In: What action should the robot take to"
    mask = torch.ones_like(ori_embeds)
    for b in range(mask.shape[0]):
        mask[b, :prefix[b], :] = 0
    mask[:, -3:, :] = 0
    return mask

def decode_embeds_to_text(adv_embeds, tokenizer, embedding_layer):
    """
    将embedding（如adv_embeds）逆向为文本。
    输入：
        adv_embeds: torch.Tensor, shape [seq_len, hidden_dim] 或 [batch, seq_len, hidden_dim]
        tokenizer: 分词器
        embedding_layer: embedding层（如model.language_model.get_input_embeddings()）
    输出：
        text: str 或 List[str]
    """
    
    # 支持batch或单条
    if adv_embeds.dim() == 2:
        adv_embeds = adv_embeds.unsqueeze(0)  # [1, seq_len, hidden_dim]
    batch_size, seq_len, hidden_dim = adv_embeds.shape
    embedding_matrix = embedding_layer.weight.data  # [vocab_size, hidden_dim]
    texts = []
    for b in range(batch_size):
        tokens = []
        for i in range(seq_len):
            emb = adv_embeds[b, i]  # [hidden_dim]
            sim = F.cosine_similarity(emb.unsqueeze(0), embedding_matrix, dim=1)  # [vocab_size]
            idx = torch.argmax(sim).item()
            tokens.append(idx)
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        texts.append(text)
    if len(texts) == 1:
        return texts[0]
    return texts

def get_synonym(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name().lower() != word.lower():
                cleaned = lemma.name().replace('_', ' ')
                cleaned_new = cleaned.translate(str.maketrans('', '', string.punctuation))
                synonyms.add(cleaned_new)
    return list(synonyms)
    
def inverse_norm(normed_6ch: torch.Tensor,
                 mean_a=(0.484375, 0.455078125, 0.40625), std_a=(0.228515625, 0.2236328125, 0.224609375),
                 mean_b=(0.5,0.5,0.5),                    std_b=(0.5,0.5,0.5)):
    """
    输入: 已归一化的 6 通道张量 (B,6,H,W)
    """

    # 1) 取后 3 通道 (branch B) 反归一化到 [0,1]
    img_b = normed_6ch[:, -3:, :, :]
    mean_b_t = torch.tensor(mean_b, device=img_b.device, dtype=img_b.dtype).view(1, 3, 1, 1)
    std_b_t  = torch.tensor(std_b,  device=img_b.device, dtype=img_b.dtype).view(1, 3, 1, 1)
    pixels   = img_b * std_b_t + mean_b_t

    return pixels


def set_norm(pixels_3ch: torch.Tensor,
             mean_a=(0.484375, 0.455078125, 0.40625), std_a=(0.228515625, 0.2236328125, 0.224609375),
             mean_b=(0.5,0.5,0.5),                    std_b=(0.5,0.5,0.5)):
    """
    输入: 已归一化的 6 通道张量 (B,6,H,W)
    """
    normalize_a = transforms.Normalize(mean=mean_a, std=std_a)
    normalize_b = transforms.Normalize(mean=mean_b, std=std_b)

    norm_a = normalize_a(pixels_3ch)
    norm_b = normalize_b(pixels_3ch)

    out = torch.cat([norm_a, norm_b], dim=1)

    return out


def add_noise_to_prismatic(normed_6ch: torch.Tensor,
                           eps: float,
                           mean_a=(0.484375, 0.455078125, 0.40625), std_a=(0.228515625, 0.2236328125, 0.224609375),
                           mean_b=(0.5,0.5,0.5),                    std_b=(0.5,0.5,0.5)):
    """
    输入: 已归一化的 6 通道张量 (B,6,H,W)
    输出: 加完噪声、重新归一化后的 6 通道张量
    """

    # 1) 取后 3 通道 (branch B) 反归一化到 [0,1]
    img_b = normed_6ch[:, -3:, :, :]
    mean_b_t = torch.tensor(mean_b, device=img_b.device, dtype=img_b.dtype).view(1, 3, 1, 1)
    std_b_t  = torch.tensor(std_b,  device=img_b.device, dtype=img_b.dtype).view(1, 3, 1, 1)
    pixels   = img_b * std_b_t + mean_b_t

    # 2) 像素空间加噪声并截断
    pixels = pixels + torch.empty_like(pixels).uniform_(-eps, eps)
    pixels = torch.clamp(pixels, min=0, max=1).detach()

    # 3) 分别归一化得到 A/B 分支
    norm_b = (pixels - mean_b_t) / std_b_t

    mean_a_t = torch.tensor(mean_a, device=pixels.device, dtype=pixels.dtype).view(1, 3, 1, 1)
    std_a_t  = torch.tensor(std_a,  device=pixels.device, dtype=pixels.dtype).view(1, 3, 1, 1)
    norm_a = (pixels - mean_a_t) / std_a_t

    out = torch.cat([norm_a, norm_b], dim=1)

    return out

@torch.no_grad()
def predict_action_batch(
    end_token_id, model, inputs, unnorm_key: str = None, **kwargs
) -> np.ndarray:
    """
    批量预测动作，返回(batch, action_dim)的动作数组
    model: 需要有generate、get_action_dim、get_action_stats、bin_centers、vocab_size等属性的方法
    """
    
    # # 1.没有 29871 的样本补 token
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    # if input_ids is not None:
    #     need_pad = input_ids[:, -1] != 29871  # (batch,)
    #     if need_pad.any():
    #         pad = torch.full((input_ids.size(0), 1), 29871, dtype=input_ids.dtype, device=input_ids.device)
    #         input_ids = torch.cat([input_ids, pad], dim=1)

    #         mask_pad = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)
    #         attention_mask = torch.cat([attention_mask, mask_pad], dim=1)
            
    #         input_ids[~need_pad, -1] = input_ids[~need_pad, -2]
 
    output = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=inputs["pixel_values"])
    next_token_logits = output.logits[:, -1, :]          # (batch, vocab_size)
    next_token_id = torch.argmax(next_token_logits, dim=-1)

    is_eos = (next_token_id == end_token_id)
    # print(next_token_id, "!!!!!")
    eos_count = is_eos.sum().item()
    
    # # 2. 批量生成
    # max_new_tokens = model.get_action_dim(unnorm_key)
    # generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, **kwargs)  # (batch, seq_len+action_dim)

    # # 3. 提取动作token
    # action_dim = model.get_action_dim(unnorm_key)
    # predicted_action_token_ids = generated_ids[:, -action_dim:].cpu().numpy()  # (batch, action_dim)

    # # 4. 离散化、归一化
    # discretized_actions = model.vocab_size - predicted_action_token_ids
    # discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1)
    # normalized_actions = model.bin_centers[discretized_actions]  # (batch, action_dim)

    # # 5. 反归一化
    # action_norm_stats = model.get_action_stats(unnorm_key)
    # mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    # action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    # actions = np.where(
    #     mask,
    #     0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
    #     normalized_actions,
    # )

    return eos_count