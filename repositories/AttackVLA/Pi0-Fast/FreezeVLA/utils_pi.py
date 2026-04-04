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
    formatted_prompts = [f"{prompt}." for prompt in prompts]
    # formatted_ref_prompts = [[f"{p}" for p in prompt_list] for prompt_list in ref_prompts]
    formatted_ref_prompts = [
        [p.replace("What action should the robot take to ", "").replace("?", ".") for p in prompt_list]
        for prompt_list in ref_prompts
    ]
    return formatted_prompts, images, formatted_ref_prompts


def get_logger(args):
    log_filename = f"log/{args.dataset}/run_pi_{args.attack}_{args.dataset}_{int(args.eps * 255)}.log"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logger = logging.getLogger("pi_logger")  # 固定logger名字，防止多进程/多线程混乱
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
    Implement cross-prompt adversarial attack on the same image with multiple prompts.
    The optimization objective is to perturb the output under all prompts.
    """
    model.eval()
    device = model.device
    model = model.module if hasattr(model, 'module') else model

    with torch.no_grad():
        ori_inputs = processor(images=[image]*len(prompt_list), text=prompt_list, return_tensors="pt", padding=True).to(torch.bfloat16).to(device)
        ori_images = inverse_norm(ori_inputs["pixel_values"][0].squeeze(0).clone().detach())

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

                # openvl does not support inputs_embeds throughout, if using input_ids it will override inputs_embeds, remember to modify
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

                # detach and save without gradients to prevent retaining the entire computation graph
                per_prompt_loss = eos_probs_text.squeeze(1).detach()
                loss_text = torch.sum(eos_probs_text)

                # Print the max index of probs
                # print("Max Index:", torch.argmax(probs_text, dim=-1), "Text Loss:", loss_text)
                
                # Calculate gradients
                grad_text = torch.autograd.grad(loss_text, adv_embeds, retain_graph=False, create_graph=False)[0]
                grad_text_norm = torch.norm(grad_text * mask, dim=-1)   # [B, seq_len]
                max_grad_idx = torch.argmax(grad_text_norm, dim=1)      # [B]

                adv_embeds = adv_embeds.detach()
                # Clear GPU memory (including large tensors related to outputs_text)
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

                        # Use regex tokenization, preserve hyphens and special quotes
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
                        
                        # Replace select_word with synonyms (preserve original punctuation and format)
                        synonyms = get_synonym(select_word)

                        if synonyms:
                            synonym = random.choice(synonyms)
                            # print(select_word, synonym)
                            # print("ori prompt: ", prompt_list[b])
                            pattern = re.compile(r'\b{}\b'.format(re.escape(select_word)), re.IGNORECASE)
                            new_prompt_list[b] = pattern.sub(synonym, prompt_list[b], count=1)
                            # print("new prompt: ", new_prompt_list[b])

                    # After replacing prompt_list, regenerate ori_inputs, ori_embeds and mask to ensure shape consistency
                    inputs_temp = processor(images=[image]*len(new_prompt_list), text=new_prompt_list, return_tensors="pt", padding=True).to(torch.bfloat16).to(device)
                    inputs_temp["pixel_values"] = set_norm(adv_images).repeat(len(new_prompt_list), 1, 1, 1)
                    outputs_temp = model(**inputs_temp, return_dict=True)
                    logits_text_temp = outputs_temp.logits
                    shift_logits_text_temp = logits_text_temp[:, -1 :, :]
                    probs_text_temp = F.log_softmax(shift_logits_text_temp, dim=-1)
                    eos_probs_text_temp = probs_text_temp[..., end_token_id]
                    per_prompt_loss_temp = eos_probs_text_temp.squeeze(1)
                    
                    # Compare and update prompt_list
                    for j in range(len(prompt_list)):
                        if per_prompt_loss_temp[j] <= per_prompt_loss[j]:
                            prompt_list[j] = new_prompt_list[j]
                            # print(f"New Loss: {per_prompt_loss_temp[j]}, Ori Loss: {per_prompt_loss[j]}")
                            # print(f"Prompt[{j}] changed from {prompt_list[j]} to {new_prompt_list[j]}")

                    # Generate ori_inputs, ori_embeds, mask, adv_embeds with the latest prompt_list
                    ori_inputs = processor(images=[image]*len(prompt_list), text=prompt_list, return_tensors="pt", padding=True).to(torch.bfloat16).to(device)
                    ori_embeds = llm(ori_inputs["input_ids"]).detach()
                    adv_embeds = ori_embeds.clone().detach()  # Ensure adv_embeds shape synchronization
                    mask = make_mask(ori_embeds, ori_inputs, model, processor)

                    # Immediately delete temporary variables after use to save GPU memory
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

        # Release GPU memory
        del outputs_image, logits_image, shift_logits_image, probs_image, eos_probs_image, loss_image, grad_img, adv_image_inputs
        torch.cuda.empty_cache()
        gc.collect()
        
    return set_norm(adv_images).to(device)

def make_mask(ori_embeds, ori_inputs, model, processor):
    pad_token_id = processor.tokenizer.pad_token_id
    pad_counts = (ori_inputs["input_ids"] == pad_token_id).sum(dim=1)
    prefix = pad_counts + 1 # 8 is the length of "What action should the robot take to"
    mask = torch.ones_like(ori_embeds)
    for b in range(mask.shape[0]):
        mask[b, :prefix[b], :] = 0
    mask[:, -1:, :] = 0
    return mask

def decode_embeds_to_text(adv_embeds, tokenizer, embedding_layer):
    """
    Reverse embedding (e.g., adv_embeds) to text.
    Input:
        adv_embeds: torch.Tensor, shape [seq_len, hidden_dim] or [batch, seq_len, hidden_dim]
        tokenizer: tokenizer
        embedding_layer: embedding layer (e.g., model.language_model.get_input_embeddings())
    Output:
        text: str or List[str]
    """
    
    # Support batch or single item
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

def inverse_norm(normed_3ch: torch.Tensor,
                 mean_b=(0.5,0.5,0.5),                    std_b=(0.5,0.5,0.5)):
    """
    Input: normalized 3-channel tensor (B,3,H,W)
    """

    # 1) Take the last 3 channels (branch B) and denormalize to [0,1]
    img_b = normed_3ch
    mean_b_t = torch.tensor(mean_b, device=img_b.device, dtype=img_b.dtype).view(1, 3, 1, 1)
    std_b_t  = torch.tensor(std_b,  device=img_b.device, dtype=img_b.dtype).view(1, 3, 1, 1)
    pixels   = img_b * std_b_t + mean_b_t

    return pixels


def set_norm(pixels_3ch: torch.Tensor,
             mean_b=(0.5,0.5,0.5),                    std_b=(0.5,0.5,0.5)):
    """
    Input: normalized 3-channel tensor (B,3,H,W)
    """
    normalize_b = transforms.Normalize(mean=mean_b, std=std_b)

    norm_b = normalize_b(pixels_3ch)

    return norm_b

def add_noise_to_prismatic(normed_3ch: torch.Tensor,
                           eps: float,
                           mean_b=(0.5,0.5,0.5),                    std_b=(0.5,0.5,0.5)):
    """
    Input: normalized 3-channel tensor (B,3,H,W)
    Output: 3-channel tensor after adding noise and renormalization
    """

    # 1) Take the last 3 channels (branch B) and denormalize to [0,1]
    img_b = normed_3ch
    mean_b_t = torch.tensor(mean_b, device=img_b.device, dtype=img_b.dtype).view(1, 3, 1, 1)
    std_b_t  = torch.tensor(std_b,  device=img_b.device, dtype=img_b.dtype).view(1, 3, 1, 1)
    pixels   = img_b * std_b_t + mean_b_t

    # 2) Add noise in pixel space and clamp
    pixels = pixels + torch.empty_like(pixels).uniform_(-eps, eps)
    pixels = torch.clamp(pixels, min=0, max=1).detach()

    # 3) Normalize separately to get A/B branches
    norm_b = (pixels - mean_b_t) / std_b_t

    return norm_b

@torch.no_grad()
def predict_action_batch(
    end_token_id, model, inputs, **kwargs
) -> np.ndarray:
    """
    Batch predict actions, return action array of shape (batch, action_dim)
    model: needs methods with attributes like generate, get_action_dim, get_action_stats, bin_centers, vocab_size
    """
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    output = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=inputs["pixel_values"])
    next_token_logits = output.logits[:, -1, :]          # (batch, vocab_size)
    next_token_id = torch.argmax(next_token_logits, dim=-1)

    is_eos = (next_token_id == end_token_id)
    # print(next_token_id, "!!!!!")
    eos_count = is_eos.sum().item()

    return eos_count