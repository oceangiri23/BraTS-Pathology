
import torch
from PIL import Image
import os
from tqdm import tqdm
import json
import csv
from transformers import AutoModelForImageTextToText, AutoProcessor

from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

model_id = "google/medgemma-4b-it"
prompt = "Describe the most dominant histopathological features visible in this image.\n\n\
                Base your descriptions on visible evidence such as cell morphology, tissue architecture, staining patterns, and structural abnormalities.\n\n\
                Do **not** use or mention any class labels or predefined terms. Just describe what is visually present and why it is significant."
system_instruction = "You are an expert neuropathologist with experience in histopathological analysis of glioblastoma tissue sections."
max_new_tokens = 200
batch_size = 64  # adjust based on your GPU memory
# _classes = ['CT', 'DM', 'IC', 'LI', 'MP', 'NC', 'PL', 'PN', 'WM']
# _classes = ["DM", "LI", "MP", "PL", "PN" ]

num_images_per_class = 100 

class_names = ['CT', 'DM', 'IC', 'LI', 'MP', 'NC', 'PL', 'PN', 'WM']
base_dir = "../BraTS-Path/New-Data-384-Collated-JPG"
output_csv = "internal_val_with_captions.csv"


_class = "WM"
image_dir = f"../BraTS-Path/New-Data-384-Collated-JPG/{_class}"
op_file = f"results_batch_{_class}_fast.jsonl"




model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)


image_paths = []
for cls in class_names:
    folder = os.path.join(base_dir, cls)
    imgs = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".jpg")
    ])[:num_images_per_class]
    image_paths.extend(imgs)

print(f"Total images to process: {len(image_paths)}")

results = []


for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing Batches"):
    batch_paths = image_paths[i:i+batch_size]
    batch_messages = []
    for image_path in batch_paths:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_instruction}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image}
                ]
            }
        ]
        batch_messages.append(messages)

    inputs = processor.apply_chat_template(
        batch_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_dict=True
    ).to(model.device, dtype=torch.bfloat16)

    input_ids = inputs["input_ids"]
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    input_lens = (input_ids != processor.tokenizer.pad_token_id).sum(dim=1)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
    
    decoded = processor.batch_decode(outputs[:, input_lens[0]:], skip_special_tokens=True)

    for path, caption in zip(batch_paths, decoded):
        results.append([path, caption.strip()])



with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "response"])
    writer.writerows(results)

print(f"\n Done! Captions saved to: {output_csv}")