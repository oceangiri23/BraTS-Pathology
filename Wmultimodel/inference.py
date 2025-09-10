# inference.py

import os
import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import timm

from dataloaders.dataset_infer import MultiModalCancerDatasetInfer as MultiModalCancerDataset
from dataloaders.collate_infer import multimodal_collate_fn_infer
from models.cross_attention_transformer import CrossAttentionTransformer
from dataloaders.augmentation import get_resize_only_transform
from provgigapath import ProvGigaPath
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
login(token=HF_TOKEN)
wandb.login(key=WANDB_API_KEY)

ckpt_path = "../provgigapath/model/provgigapath_finetuned.ckpt"

def load_finetuned_encoder_only(ckpt_path, num_classes=9):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint['state_dict']

    model = ProvGigaPath(num_classes=num_classes, checkpoint_path=None)

    encoder_state_dict = {
        k.replace("model.encoder.", ""): v
        for k, v in state_dict.items()
        if k.startswith("model.encoder.")
    }

    model.encoder.load_state_dict(encoder_state_dict, strict=False)
    return model.encoder

def predict_and_save(config_path, checkpoint_path, output_csv_path, device):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-large")

    eval_dataset = MultiModalCancerDataset(
        csv_path=config["val_csv_path"],
        image_root=config["image_root"],
        tokenizer=tokenizer,
        transform=get_resize_only_transform(),  # Resize to 224x224
    )

    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=multimodal_collate_fn_infer,
    )

    # Encoders
    #image_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    #image_encoder.eval().to(device).requires_grad_(False)

    image_encoder = load_finetuned_encoder_only(ckpt_path)
    image_encoder.eval().to(device)
    image_encoder.requires_grad_(False)

    text_encoder = AutoModel.from_pretrained("michiyasunaga/BioLinkBERT-large")
    text_encoder.eval().to(device).requires_grad_(False)
    
    fusion_model = CrossAttentionTransformer(
        image_embedding_dim=config["image_embedding_dim"],
        text_embedding_dim=config["text_embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_heads=config["num_attention_heads"],
        num_layers=config["num_transformer_layers"],
        num_classes=config["num_classes"],
        dropout=config["dropout"]
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    fusion_model.load_state_dict(checkpoint["model_state_dict"])
    #fusion_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    fusion_model.eval()

    
    all_preds = []
    image_paths = []

    print("Inference started...")
    m = 0
    with torch.no_grad():
        for batch in eval_loader:
            m += 1
            print(f"Processing batch {m}/{len(eval_loader)}")
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            image_embeds = image_encoder(images)
            text_outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_embeds = text_outputs.last_hidden_state

            logits = fusion_model(image_embeds, text_embeds, attention_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            image_paths.extend(batch["image"]) 

    
    input_df = pd.read_csv(config["val_csv_path"])
    input_df = input_df.reset_index(drop=True)
    input_df["Prediction"] = all_preds
    output_df = input_df[["image", "Prediction"]]
    output_df.to_csv(output_csv_path, index=False)
    print(f" Predictions saved to: {output_csv_path}")


if __name__ == "__main__":

    config_path = "config/config.yaml"
    checkpoint_path = "data/checkpoints/withfunetunedimagencoder/multimodel_classifier.pt"
    output_csv_path = "data/NEW_CSV/Prediction6_balanceclass_3000.csv"
    device = "cuda"

    predict_and_save(config_path,checkpoint_path, output_csv_path, device)
