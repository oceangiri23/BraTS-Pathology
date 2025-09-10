
import torch
from torch.nn.utils.rnn import pad_sequence

def multimodal_collate_fn_infer(batch):

    images = torch.stack([item["image"] for item in batch])  # [B, C, H, W]
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    image_paths = [item["image"] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    return {
        "image": images,
        "input_ids": input_ids_padded,
        "attention_mask": attention_masks_padded,
        "image_path": image_paths
    }
