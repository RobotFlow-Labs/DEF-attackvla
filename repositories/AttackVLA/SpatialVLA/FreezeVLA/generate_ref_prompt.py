import os
import io
import json
import torch
import random
import openai
import base64
import logging
import argparse
from tqdm import tqdm
from PIL import Image
from tenacity import retry, wait_fixed
import re

class GPT:
    def __init__(self, model):
        self.client = openai.OpenAI(
            api_key="xxx",
            base_url="xxx",
        )
        self.model = model

    @retry(wait=wait_fixed(10))
    def __call__(self, uprompt, history=[]) -> str:
        messages = []
        messages.extend([{"role": "user", "content": h[0]}, {"role": "assistant", "content": h[1]}] for h in history)
        messages.extend(uprompt if isinstance(uprompt,list) else [{"role": "user", "content": uprompt}])
        try:
            completion = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=2
            )
            response = completion.choices[0].message.content
            return response
        except Exception as e:
            logging.error(f"{e}")
            raise

def image_to_base64(image):
    if isinstance(image, bytes):
        img_bytes = image
    elif hasattr(image, 'save'):
        buf = io.BytesIO()
        image.save(buf, format='JPEG')
        img_bytes = buf.getvalue()
    else:
        img = Image.fromarray(image)
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        img_bytes = buf.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')

def image_to_ref_prompts(image, gpt, num_prompts=20):
    base64_image = image_to_base64(image)
    system_prompt = (
        "You are an expert robot task planning assistant. "
        "Given an image, analyze the scene and generate a list of clear, concise, high-quality reference prompts describing different specific actions the robot could take. "
        "Focus on actionable, unambiguous instructions suitable for downstream robot planning. "
        "Do not include unnecessary information or speculation. "
        f"Output exactly {num_prompts} imperative English sentences, each using the template: 'What action should the robot take to {{prompt}}?' "
        "where {prompt} is a concise description of the goal or task in the image. "
        f"Number each prompt from 1 to {num_prompts}. "
        "If the image does not contain enough obvious actions, please use your imagination to invent plausible actions that a robot could perform in this scene. "
        "Do not repeat similar actions; make each prompt as unique as possible. "
        "Please ensure that the {prompts} do not contain any special symbols or punctuation marks, such as commas, dashes, colons, or any other punctuation."
    )
    user_content = [
        {"type": "text", "text": (
            f"Based on the image, generate {num_prompts} high-quality, diverse reference prompts that clearly describe different specific actions the robot could perform. "
            "If the image content is limited, please use your imagination to create more possible actions. "
            "Be precise and concise. Output as a numbered list.")},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    ]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    prompt_text = gpt(messages)
    prompts = []
    for line in prompt_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line[0].isdigit():
            line = line.lstrip("0123456789. ").strip()
        elif line.startswith('-'):
            line = line[1:].strip()
        if line:
            line = re.sub(r'[^\w\s]', '', line) + '?'
            prompts.append(line)

    while len(prompts) < num_prompts:
        prompts.append("Error: Not enough prompts returned.")
    return prompts[:num_prompts]

def generate_ref_prompts(gpt, dataset_path, max_sample=256, num_prompts=20, save_dir=None):
    save_prompt = os.path.join(save_dir, 'ref_prompts.json')
    save_image = os.path.join(save_dir, 'images')
    save_dataset = os.path.join(save_dir, 'libero_10_no_noops_sampled.pt')
    os.makedirs(save_image, exist_ok=True)
    dataset = torch.load(dataset_path, weights_only=False)
    try:
        dataset = random.sample(dataset, max_sample)
    except Exception as e:
        raise ValueError(f"Error: {e}")
    

    results = []
    new_dataset = []
    for idx, item in enumerate(tqdm(dataset, desc="Generating ref prompts")):
        image = item['images'][0]
        img_filename = f"img_{idx:04d}.jpg"
        img_path = os.path.join(save_image, img_filename)
        if idx < 5:
            image.save(img_path, format='JPEG')

        try:
            ref_prompts = image_to_ref_prompts(image, gpt, num_prompts=num_prompts)
        except Exception as e:
            ref_prompts = [f"Error: {e}"] * num_prompts
        results.append({'image_file': img_filename, 'ref_prompts': ref_prompts, 'prompt': item['text']})

        new_item = {
            'image': item['images'],
            'prompt': item['text'],
            'ref_prompts': ref_prompts
        }
        new_dataset.append(new_item)

    with open(save_prompt, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"All prompts saved to {save_prompt}")

    torch.save(new_dataset, save_dataset)
    print(f"Merged dataset with ref_prompts saved to {save_dataset}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='bridge_orig_val', help='Directory to save all outputs')
    parser.add_argument('--dataset_path', type=str, default='bridge_orig_val.pt', help='Path to dataset .pt file')
    parser.add_argument('--max_sample', type=int, default=300, help='Number of random samples to generate ref prompts for')
    parser.add_argument('--num_prompts', type=int, default=25, help='Number of prompts to generate per image')
    args = parser.parse_args()

    generate_ref_prompts(
        GPT("o3"),
        dataset_path = os.path.join('/path/to/dataset/folder/dataset/', args.dataset_path),
        max_sample=args.max_sample,
        num_prompts=args.num_prompts,
        save_dir=os.path.join('/path/to/dataset/folder/dataset/gpt/', args.save_dir)
    )