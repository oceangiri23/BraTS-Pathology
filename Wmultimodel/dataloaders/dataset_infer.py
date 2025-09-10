
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from .augmentation import apply_transform  

class MultiModalCancerDatasetInfer(Dataset):
    def __init__(self, 
                 csv_path=None, 
                 data_frame=None,
                 image_root=None, 
                 tokenizer=None, 
                 transform=None,            
                 max_length=128):
     
        if csv_path:
            self.data = pd.read_csv(csv_path)
        elif data_frame is not None:
            self.data = data_frame.reset_index(drop=True)
        else:
            raise ValueError("Provide either csv_path or dataframe.")
        
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

        self.post_transform = transforms.Compose([
            #transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                 std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        relative_image_path = row['image'].lstrip('/')
        image_path = os.path.join(self.image_root, relative_image_path)
        image = np.array(Image.open(image_path).convert("RGB"))

        if self.transform:
            image = apply_transform(image, self.transform)

        image = image / 255.0
        image = torch.tensor(image).permute(2, 0, 1).float()

        caption = str(row["response"])
        encoding = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "image": image,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "image_path": row["image"]  # for result saving
        }
