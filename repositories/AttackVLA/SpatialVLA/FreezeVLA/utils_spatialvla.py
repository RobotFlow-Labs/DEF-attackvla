import os
import copy
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
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

def custom_collate_fn(batch):
    prompts = [item[0] for item in batch]
    images = [item[1] for item in batch]
    formatted_prompts = [f"What action should the robot take to {prompt}?" for prompt in prompts]
    return formatted_prompts, images

def custom_collate_fn_with_ref(batch):
    prompts = [item[0] for item in batch]
    images = [item[1] for item in batch]
    ref_prompts = [item[2] for item in batch]
    formatted_prompts = [f"What action should the robot take to {prompt}?" for prompt in prompts]
    return formatted_prompts, images, ref_prompts

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
    log_filename = f"log/{args.dataset}/run_spatialvla_{args.attack}_{args.image_step}_{args.text_step}.log"
    logger = logging.getLogger("spatialvla_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False

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

def one_prompt_attack(model, processor, images, prompts, epsilon=4/255, alpha=1/255, step=100, random_start=True):
    model.eval()
    device = model.device

    eos_token_id = processor.tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("EOS token ID could not be determined. Cannot proceed with the attack.")

    # print(model.vocab_size, eos_token_id)

    ori_inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True).to(torch.bfloat16).to(device)
    ori_images = ori_inputs["pixel_values"].clone().detach()
    input_len  = ori_inputs["input_ids"].shape[-1]

    adv_inputs = copy.deepcopy(ori_inputs).to(torch.bfloat16).to(device)
    adv_images = adv_inputs["pixel_values"].clone().detach().requires_grad_(True)

    # Starting at a uniformly random point
    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        adv_inputs["pixel_values"] = adv_images
        
    for _ in range(step):
        adv_images.requires_grad = True
        outputs = model(**adv_inputs, return_dict=True)

        # Calculate loss
        logits = outputs.logits
        shift_logits = logits[:, input_len - 1 :, :]
        
        probs = F.log_softmax(shift_logits, dim=-1)
        eos_probs = probs[..., eos_token_id]
        loss = torch.sum(eos_probs)

        # Print the max index of probs
        # print("Max Index:", torch.argmax(probs, dim=-1), "Loss:", loss)

        # Update adversarial images
        grad       = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
        adv_images = adv_images.detach() + alpha * grad.sign()
        delta      = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(ori_images + delta, min=0, max=1).detach()
        
        adv_inputs["pixel_values"] = adv_images

    return adv_inputs, adv_images

def multi_prompt_attack(model, processor, image, prompt_list, epsilon=4/255, alpha=1/255, step=100, random_start=True):
    model.eval()
    device = model.device
    model = model.module if hasattr(model, 'module') else model

    eos_token_id = processor.tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("EOS token ID could not be determined. Cannot proceed with the attack.")

    with torch.no_grad():
        ori_inputs = processor(images=[image]*len(prompt_list), text=prompt_list, return_tensors="pt", padding=True).to(torch.bfloat16).to(device)
        ori_images = ori_inputs["pixel_values"][0].squeeze(0).clone().detach()
    adv_images = ori_images.clone().detach()

    if random_start:
        # Starting at a uniformly random point
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(step):
        adv_images.requires_grad = True
        adv_inputs = {
            "input_ids": ori_inputs['input_ids'],
            "attention_mask": ori_inputs['attention_mask'],
            "pixel_values": adv_images.repeat(len(prompt_list), 1, 1, 1),
            "intrinsic": ori_inputs['intrinsic']
        }

        # Calculate loss
        outputs = model(**adv_inputs, return_dict=True)
        logits = outputs.logits
        shift_logits = logits[:, -1 :, :]
        probs = F.log_softmax(shift_logits, dim=-1)
        eos_probs = probs[..., eos_token_id]
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

    return adv_images

def stop_action_embedding_attack(model, processor, image, prompt_list, text_step=10, epsilon=4/255, alpha=1/255, image_step=100, random_start=True):
    model.eval()
    device = model.device
    model = model.module if hasattr(model, 'module') else model

    eos_token_id = processor.tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("EOS token ID could not be determined. Cannot proceed with the attack.")

    with torch.no_grad():
        ori_inputs = processor(images=[image]*len(prompt_list), text=prompt_list, return_tensors="pt", padding=True).to(torch.bfloat16).to(device)
        ori_images = ori_inputs["pixel_values"][0].squeeze(0).clone().detach()
    adv_images = ori_images.clone().detach()

    llm = model.language_model.get_input_embeddings()
    ori_embeds = llm(ori_inputs["input_ids"]).detach()
    adv_embeds = ori_embeds.clone().detach()

    pad_token_id = processor.tokenizer.pad_token_id
    pad_counts = (ori_inputs["input_ids"] == pad_token_id).sum(dim=1)
    num_image_tokens = model.config.text_config.num_image_tokens
    prefix = pad_counts + num_image_tokens + 8 # 8 is the length of "What action should the robot take to"
    mask = torch.ones_like(ori_embeds)
    for b in range(mask.shape[0]):
        mask[b, :prefix[b], :] = 0
    mask[:, -2:, :] = 0

    if random_start:
        # Starting at a uniformly random point
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for i in range(image_step):

        # optimize adversarial text embedding, find the hard prompt
        if i == 0:
            for _ in range(text_step):
                adv_embeds.requires_grad = True
                adv_text_inputs = {
                    "input_ids": ori_inputs['input_ids'],
                    "inputs_embeds": adv_embeds.clone(),
                    "attention_mask": ori_inputs['attention_mask'],
                    "pixel_values": adv_images.repeat(len(prompt_list), 1, 1, 1),
                    "intrinsic": ori_inputs['intrinsic']
                }

                # Calculate loss
                outputs_text = model(**adv_text_inputs, return_dict=True)
                logits_text = outputs_text.logits
                shift_logits_text = logits_text[:, -1 :, :]
                probs_text = F.log_softmax(shift_logits_text, dim=-1)
                eos_probs_text = probs_text[..., eos_token_id]
                loss_text = torch.sum(eos_probs_text)

                # # Print the max index of probs
                # print("Max Index:", torch.argmax(probs_text, dim=-1), "Text Loss:", loss_text)
                
                # Calculate gradients
                grad_text = torch.autograd.grad(loss_text, adv_embeds, retain_graph=False, create_graph=False)[0]

                # Update adversarial prompts
                adv_embeds = adv_embeds.detach() - 0.00005 * grad_text.sign()
                delta_text = mask * torch.clamp(adv_embeds - ori_embeds, min=-0.25, max=0.25)
                # adv_embeds = torch.clamp(ori_embeds + delta_text, min=ori_embeds.min(), max=ori_embeds.max()).detach()
                adv_embeds = (ori_embeds + delta_text).detach()

        adv_images.requires_grad = True
        adv_image_inputs = {
            "input_ids": ori_inputs['input_ids'],
            "inputs_embeds": adv_embeds.clone(),
            "attention_mask": ori_inputs['attention_mask'],
            "pixel_values": adv_images.repeat(len(prompt_list), 1, 1, 1),
            "intrinsic": ori_inputs['intrinsic']
        }

        # Calculate loss
        outputs_image = model(**adv_image_inputs, return_dict=True)
        logits_image = outputs_image.logits
        shift_logits_image = logits_image[:, -1 :, :]
        probs_image = F.log_softmax(shift_logits_image, dim=-1)
        eos_probs_image = probs_image[..., eos_token_id]
        loss_image = torch.sum(eos_probs_image)

        # Print the max index of probs
        # print("Max Index:", torch.argmax(probs_image, dim=-1), "Image Loss:", loss_image)

        # Calculate gradients
        grad_img = torch.autograd.grad(loss_image, adv_images, retain_graph=False, create_graph=False)[0]

        # Update adversarial images
        adv_images = adv_images.detach() + alpha * grad_img.sign()
        delta_img = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(ori_images + delta_img, min=0, max=1).detach()

    return adv_images


def stop_action_token_attack(model, processor, image, prompt_list, text_step=10, epsilon=4/255, alpha=1/255, image_step=100, random_start=True):
    model.eval()
    device = model.device
    model = model.module if hasattr(model, 'module') else model

    eos_token_id = processor.tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("EOS token ID could not be determined. Cannot proceed with the attack.")

    with torch.no_grad():
        ori_inputs = processor(images=[image]*len(prompt_list), text=prompt_list, return_tensors="pt", padding=True).to(torch.bfloat16).to(device)
        ori_images = ori_inputs["pixel_values"][0].squeeze(0)  # 只clone或detach一次即可
    adv_images = ori_images.clone()

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
                adv_text_inputs = {
                    "input_ids": ori_inputs['input_ids'],
                    "inputs_embeds": adv_embeds.clone(),
                    "attention_mask": ori_inputs['attention_mask'],
                    "pixel_values": adv_images.repeat(len(prompt_list), 1, 1, 1),
                    "intrinsic": ori_inputs['intrinsic']
                }

                # Calculate loss
                outputs_text = model(**adv_text_inputs, return_dict=True)
                logits_text = outputs_text.logits
                shift_logits_text = logits_text[:, -1 :, :]
                probs_text = F.log_softmax(shift_logits_text, dim=-1)
                eos_probs_text = probs_text[..., eos_token_id]

                per_prompt_loss = eos_probs_text.squeeze(1)
                loss_text = torch.sum(eos_probs_text)

                # Print the max index of probs
                # print("Max Index:", torch.argmax(probs_text, dim=-1), "Text Loss:", loss_text)
                
                # Calculate gradients
                grad_text = torch.autograd.grad(loss_text, adv_embeds, retain_graph=False, create_graph=False)[0]
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
                        
                        synonyms = get_synonym(select_word)

                        if synonyms:
                            synonym = random.choice(synonyms)
                            pattern = re.compile(r'\b{}\b'.format(re.escape(select_word)), re.IGNORECASE)
                            new_prompt_list[b] = pattern.sub(synonym, prompt_list[b], count=1)

                    inputs_temp = processor(images=[image]*len(new_prompt_list), text=new_prompt_list, return_tensors="pt", padding=True).to(torch.bfloat16).to(device)
                    inputs_temp["pixel_values"] = adv_images.repeat(len(new_prompt_list), 1, 1, 1)
                    outputs_temp = model(**inputs_temp, return_dict=True)
                    logits_text_temp = outputs_temp.logits
                    shift_logits_text_temp = logits_text_temp[:, -1 :, :]
                    probs_text_temp = F.log_softmax(shift_logits_text_temp, dim=-1)
                    eos_probs_text_temp = probs_text_temp[..., eos_token_id]
                    per_prompt_loss_temp = eos_probs_text_temp.squeeze(1)
                    
                    for j in range(len(prompt_list)):
                        if per_prompt_loss_temp[j] <= per_prompt_loss[j]:
                            prompt_list[j] = new_prompt_list[j]
                            # print(f"New Loss: {per_prompt_loss_temp[j]}, Ori Loss: {per_prompt_loss[j]}")
                            # print(f"Prompt[{j}] changed from {prompt_list[j]} to {new_prompt_list[j]}")

                    ori_inputs = processor(images=[image]*len(prompt_list), text=prompt_list, return_tensors="pt", padding=True).to(torch.bfloat16).to(device)
                    ori_embeds = llm(ori_inputs["input_ids"]).detach()
                    adv_embeds = ori_embeds.clone().detach()  # 保证adv_embeds形状同步
                    mask = make_mask(ori_embeds, ori_inputs, model, processor)

                    torch.cuda.empty_cache()

        adv_images.requires_grad = True
        adv_image_inputs = {
            "input_ids": ori_inputs['input_ids'],
            "attention_mask": ori_inputs['attention_mask'],
            "pixel_values": adv_images.repeat(len(prompt_list), 1, 1, 1),
            "intrinsic": ori_inputs['intrinsic']
        }

        # Calculate loss
        outputs_image = model(**adv_image_inputs, return_dict=True)
        logits_image = outputs_image.logits
        shift_logits_image = logits_image[:, -1 :, :]
        probs_image = F.log_softmax(shift_logits_image, dim=-1)
        eos_probs_image = probs_image[..., eos_token_id]
        loss_image = torch.sum(eos_probs_image)

        # Print the max index of probs
        # print("Max Index:", torch.argmax(probs_image, dim=-1), "Image Loss:", loss_image)

        # Calculate gradients
        grad_img = torch.autograd.grad(loss_image, adv_images, retain_graph=False, create_graph=False)[0]

        # Update adversarial images
        adv_images = adv_images.detach() + alpha * grad_img.sign()
        delta_img  = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(ori_images + delta_img, min=0, max=1).detach()
            
    return adv_images

def make_mask(ori_embeds, ori_inputs, model, processor):
    pad_token_id = processor.tokenizer.pad_token_id
    pad_counts = (ori_inputs["input_ids"] == pad_token_id).sum(dim=1)
    num_image_tokens = model.config.text_config.num_image_tokens
    prefix = pad_counts + num_image_tokens + 8 # 8 is the length of "What action should the robot take to"
    mask = torch.ones_like(ori_embeds)
    for b in range(mask.shape[0]):
        mask[b, :prefix[b], :] = 0
    mask[:, -2:, :] = 0
    return mask

def get_synonym(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name().lower() != word.lower():
                cleaned = lemma.name().replace('_', ' ')
                cleaned_new = cleaned.translate(str.maketrans('', '', string.punctuation))
                synonyms.add(cleaned_new)
    return list(synonyms)