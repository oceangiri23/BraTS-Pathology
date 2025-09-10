
import os
import yaml
import torch
import wandb
import timm
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import WeightedRandomSampler
from provgigapath import ProvGigaPath

from dataloaders.dataset import MultiModalCancerDataset
from dataloaders.collate import multimodal_collate_fn
from dataloaders.augmentation import get_transform_setup_1, get_resize_only_transform
from models.cross_attention_transformer import modifiedCrossAttentionTransformer
from sklearn.model_selection import train_test_split


from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
login(token=HF_TOKEN)
wandb.login(key=WANDB_API_KEY)


with open("config/config.yaml") as f:
    config = yaml.safe_load(f)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs(config["checkpoint_dir"], exist_ok=True)


wandb.init(
    project=config["wandb_project"],
    #entity=config["wandb_entity"],
    name=config["wandb_run_name"],
    config=config
)

df = pd.read_csv(config["csv_path"])

df = df.rename(columns={
    "Input Path": "image_path",
    "Ground_Truth": "label",
    "response": "caption"
})



label_map = {
            'CT': 0, 'PN': 1, 'MP': 2, 'NC': 3, 'IC': 4, 
            'WM': 5, 'LI': 6, 'DM': 7, 'PL': 8
        }

idx_to_label = {v: k for k, v in label_map.items()}

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df["label"])
tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-large")

label_indices = train_df["label"].map(label_map)
class_counts = label_indices.value_counts().sort_index().values  # class frequency
class_weights = 1.0 / class_counts                                # inverse frequency
sample_weights = [class_weights[label] for label in label_indices]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


train_dataset = MultiModalCancerDataset(
    csv_path=None,
    data_frame = train_df,
    image_root=config["image_root"],
    tokenizer=tokenizer,
    transform=get_transform_setup_1()
)

val_dataset = MultiModalCancerDataset(
    csv_path=None,
    data_frame=val_df,
    image_root=config["image_root"],
    tokenizer=tokenizer,
    transform=get_resize_only_transform()
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=config["batch_size"],
    sampler=sampler,
    num_workers=config["num_workers"],
    collate_fn=multimodal_collate_fn
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["num_workers"],
    collate_fn=multimodal_collate_fn
)


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


ckpt_path = "../provgigapath/model/provgigapath_finetuned.ckpt"
print("Loading frozen ProGigaPath and BioLinkBERT...")
#image_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
#image_encoder.eval().to(device)
#image_encoder.requires_grad_(False)

image_encoder = load_finetuned_encoder_only(ckpt_path)
image_encoder.eval().to(device)
image_encoder.requires_grad_(False)





text_encoder = AutoModel.from_pretrained("michiyasunaga/BioLinkBERT-large")
text_encoder.eval().to(device)
text_encoder.requires_grad_(False)
print("Loading Complete...")

fusion_model = modifiedCrossAttentionTransformer(
    image_embedding_dim=config["image_embedding_dim"],
    text_embedding_dim=config["text_embedding_dim"],
    hidden_dim=config["hidden_dim"],
    num_heads=config["num_attention_heads"],
    num_layers=config["num_transformer_layers"],
    num_classes=config["num_classes"],
    dropout=config["dropout"]
).to(device)

#criterion = torch.torch.nn.BCEWithLogitsLoss()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))


best_f1 = 0.0
start_epoch = 0

checkpoint_files = [f for f in os.listdir(config["checkpoint_dir"]) if f.startswith("model_epoch_")]
if checkpoint_files:
    checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))  # Sort by epoch
    latest_checkpoint = checkpoint_files[-1]
    checkpoint_path = os.path.join(config["checkpoint_dir"], latest_checkpoint)
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    fusion_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_f1 = checkpoint.get('val_f1', 0.0)
    
    print(f"Resuming from epoch {start_epoch} with best F1 = {best_f1:.4f}")
else:
    print("No checkpoint found. Starting training from scratch.")




for epoch in range(start_epoch, config["num_epochs"]):
    print(f"\nStarting Epoch [{epoch+1}/{config['num_epochs']}]")
    fusion_model.train()
    total_train_loss = 0
    train_preds = []
    train_labels = []

    loop = tqdm(train_loader, desc=f"Training", ncols=100)

    for batch in loop:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            image_embeds = image_encoder(images)  # [B, 2048]
            text_outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_embeds = text_outputs.last_hidden_state  # [B, seq_len, 1024]

        logits = fusion_model(image_embeds, text_embeds, attention_mask)  # [B, num_classes]
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        #preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        #labels = labels.detach().cpu().numpy()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        true_labels = torch.argmax(labels, dim=1).detach().cpu().numpy()

        train_preds.extend(preds)
        train_labels.extend(true_labels)

    train_f1 = f1_score(train_labels, train_preds, average='macro')
    train_f1_per_class = f1_score(train_labels, train_preds, average=None)
    avg_train_loss = total_train_loss / len(train_loader)

    fusion_model.eval()
    total_val_loss = 0
    val_preds = []
    val_labels = []

    loop2 = tqdm(val_loader, desc=f"Validtion", ncols=100)
    with torch.no_grad():
        for batch in loop2:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            image_embeds = image_encoder(images)
            text_outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_embeds = text_outputs.last_hidden_state

            logits = fusion_model(image_embeds, text_embeds, attention_mask)
            loss = criterion(logits, labels)

            total_val_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            true_labels = torch.argmax(labels, dim=1).cpu().numpy()

            val_preds.extend(preds)
            val_labels.extend(true_labels)


  
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    val_f1_per_class = f1_score(val_labels, val_preds, average=None)
    avg_val_loss = total_val_loss / len(val_loader)


    log_data = {
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "train_f1": train_f1,
        "val_f1": val_f1,
        "epoch": epoch + 1,
    }

    for idx, f1 in enumerate(train_f1_per_class):
        log_data[f"train_f1_{idx_to_label[idx]}"] = f1

    for idx, f1 in enumerate(val_f1_per_class):
        log_data[f"val_f1_{idx_to_label[idx]}"] = f1

    wandb.log(log_data)

    print(f"Epoch [{epoch+1}] -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

    checkpoint_path = os.path.join(config["checkpoint_dir"], f"model_epoch_{epoch+1}.pt")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': fusion_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_f1': val_f1,
    }, checkpoint_path)

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_path = os.path.join(config["checkpoint_dir"], "best_model.pt")
        torch.save(fusion_model.state_dict(), best_path)