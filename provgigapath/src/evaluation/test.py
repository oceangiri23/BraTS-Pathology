### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
from typing import Union, Iterable, Any, Callable
import time
import random

### External Imports ###
import numpy as np
import torch as tc
import pandas as pd
from skimage import io
from PIL import Image

### Internal Imports ###
from paths import pc_paths as p
from networks import provgigapath
from evaluation import inference


#for training = {'WM': 0, 'CT' : 1, 'PN': 2, 'NC': 3, 'MP': 4, 'IC': 5, 'LI':6, 'PL':7, 'DM':8}
# for submission ={ 'CT' 0:, 'PN' : 1,'MP':2, 'NC': 3, 'IC': 4, 'WM': 5, 'LI': 6, 'DM': 7, 'PL': 8}

class_mapper = { 1: 0,  2: 1, 4: 2, 3: 3, 5: 4, 0: 5, 6: 6, 8: 7, 7: 8 }

def predict_folder_images(image_dir, save_path, model, device, network_transforms=None):
    image_dir = Path(image_dir)
    images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    class_mapper = {1: 0, 2: 1, 4: 2, 3: 3, 5: 4, 0: 5, 6: 6, 8: 7, 7: 8}

    results = []
    for idx, image_name in enumerate(images):
        print(f"Inferencing {idx+1}/{len(images)}: {image_name}")
        image_path = image_dir / image_name
        with tc.no_grad():
            output = inference.inference_single(image_path, model, mode='prov', device=device, network_transforms=network_transforms)
        prediction = tc.argmax(output).item()
        mapped_prediction = class_mapper[prediction]
        results.append((image_name, mapped_prediction))

    df = pd.DataFrame(results, columns=["SubjectID", "Prediction"])
    df.to_csv(save_path, index=False)
    print(f"\nSaved results to {save_path}")

def load_state_dict(checkpoint_path):
    checkpoint = tc.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    output_state_dict = {}
    for key in state_dict.keys():
        if "model.m" in key:
            output_state_dict[key.replace("model.m", "m")] = state_dict[key]
        elif "encoder" in key or "fc" in key:
            output_state_dict[key.replace("model.", "")] = state_dict[key]
    return output_state_dict

def load_provgigapath(checkpoint_path, device):
    num_classes = 9
    model = provgigapath.ProvGigaPath(num_classes=num_classes, checkpoint_path=p.provgigapath_path)
    state_dict = load_state_dict(checkpoint_path)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

def run():
    device = "cuda:0" if tc.cuda.is_available() else "cpu"
    
    image_folder = Path("../../../BraTS-Path/BraTS-Path2025-Valid-JPG-fixed-naming/Validation-Data-Anonymized")  

    save_path = p.results_path / "Tenth_submission.csv"

    
    checkpoint_path = p.checkpoints_path / "BraTS-Path_Expfirst_Fold2_Aug" / "epoch=2_f1.ckpt"
    model = load_provgigapath(checkpoint_path, device)

 
    network_transforms = provgigapath.transforms

    
    predict_folder_images(image_folder, save_path, model, device, network_transforms)

if __name__ == "__main__":
    run()
