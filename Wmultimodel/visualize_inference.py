
import os
import yaml
import timm
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from dataloaders.dataset_infer import MultiModalCancerDatasetInfer as MultiModalCancerDataset
from dataloaders.collate_infer import multimodal_collate_fn_infer
from models.visualize_cross_attention_transformers import modifiedCrossAttentionTransformer
from dataloaders.augmentation import get_resize_only_transform
from provgigapath import ProvGigaPath
from visualization_utils import AttentionVisualizer, create_visualization_dir

from huggingface_hub import login
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

def predict_and_visualize(config_path, checkpoint_path, output_csv_path, device, visualize_samples=True):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    viz_dir = create_visualization_dir("inference_visualizations")

    label_map = {
        'CT': 0, 'PN': 1, 'MP': 2, 'NC': 3, 'IC': 4, 
        'WM': 5, 'LI': 6, 'DM': 7, 'PL': 8
    }

    tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-large")

    eval_dataset = MultiModalCancerDataset(
        csv_path=config["val_csv_path"],
        image_root=config["image_root"],
        tokenizer=tokenizer,
        transform=get_resize_only_transform(),
    )

    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=multimodal_collate_fn_infer,
    )
    
    image_encoder = load_finetuned_encoder_only(ckpt_path)
    image_encoder.eval().to(device)
    image_encoder.requires_grad_(False)

    text_encoder = AutoModel.from_pretrained("michiyasunaga/BioLinkBERT-large")
    text_encoder.eval().to(device).requires_grad_(False)

    # Fusion Model
    fusion_model = modifiedCrossAttentionTransformer(
        image_embedding_dim=config["image_embedding_dim"],
        text_embedding_dim=config["text_embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_heads=config["num_attention_heads"],
        num_layers=config["num_transformer_layers"],
        num_classes=config["num_classes"],
        dropout=config["dropout"]
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    fusion_model.load_state_dict(checkpoint["model_state_dict"])
    fusion_model.eval()
    
    attention_visualizer = AttentionVisualizer(fusion_model, tokenizer, label_map)

    all_preds = []
    all_confidences = []
    image_paths = []
    
    all_attention_weights = []
    all_embeddings = []

    print("Inference with visualization started...")
    batch_count = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            batch_count += 1
            print(f"Processing batch {batch_count}/{len(eval_loader)}")
            
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            image_embeds = image_encoder(images)
            text_outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_embeds = text_outputs.last_hidden_state

            logits, attention_weights, embeddings = fusion_model(
                image_embeds, text_embeds, attention_mask, return_attention=True
            )
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            confidences = torch.softmax(logits, dim=1).max(dim=1)[0].cpu().numpy()

            all_preds.extend(preds)
            all_confidences.extend(confidences)
            image_paths.extend(batch["image"])
            
            all_attention_weights.append([aw.cpu() for aw in attention_weights])
            all_embeddings.append({
                'image': embeddings['image_embeds'].cpu(),
                'text': embeddings['text_embeds'].cpu(),
                'combined': embeddings['combined_embeds'].cpu()
            })
            
            if visualize_samples and batch_count <= 5:  # Visualize first 5 batches
                for i in range(min(2, len(images))):  # 2 samples per batch
                    sample_idx = (batch_count - 1) * config["batch_size"] + i
                    
                    img_embed = image_embeds[i:i+1]
                    txt_embed = text_embeds[i:i+1]
                    attn_mask = attention_mask[i:i+1]
                    inp_ids = input_ids[i:i+1]
                    
                    pred_label = preds[i]
                    confidence = confidences[i]
                    
                    print(f"Visualizing sample {sample_idx}: Predicted {pred_label} (confidence: {confidence:.4f})")
                    
                    for layer_idx in range(config["num_transformer_layers"]):
                        for head_idx in range(min(3, config["num_attention_heads"])):  # First 3 heads
                            save_path = f"{viz_dir}/attention_heatmaps/sample_{sample_idx}_layer_{layer_idx}_head_{head_idx}.png"
                            attention_visualizer.visualize_attention_heatmap(
                                img_embed, txt_embed, attn_mask, inp_ids,
                                true_label=None, save_path=save_path,
                                layer_idx=layer_idx, head_idx=head_idx
                            )
                    
                    save_path = f"{viz_dir}/layer_comparisons/sample_{sample_idx}_layer_comparison.png"
                    #attention_visualizer.visualize_layer_comparison(
                     #   img_embed, txt_embed, attn_mask, inp_ids, save_path=save_path
                    #)
                    
                    save_path = f"{viz_dir}/embedding_analysis/sample_{sample_idx}_similarity.png"
                    attention_visualizer.visualize_embedding_similarity(
                        img_embed, txt_embed, attn_mask, save_path=save_path
                    )
                    
                    save_path = f"{viz_dir}/attention_heatmaps/sample_{sample_idx}_rollout.png"
                    attention_visualizer.visualize_attention_rollout(
                        img_embed, txt_embed, attn_mask, inp_ids, save_path=save_path
                    )
    input_df = pd.read_csv(config["val_csv_path"])
    input_df = input_df.reset_index(drop=True)

    input_df["Prediction"] = all_preds
    input_df["Confidence"] = all_confidences

    output_df = input_df[["image", "Prediction", "Confidence"]]
    output_df.to_csv(output_csv_path, index=False)

    generate_summary_visualizations(all_preds, all_confidences, all_attention_weights, 
                                   all_embeddings, label_map, viz_dir)

    print(f"Predictions saved to: {output_csv_path}")
    print(f"Visualizations saved to: {viz_dir}")

def generate_summary_visualizations(predictions, confidences, attention_weights, embeddings, label_map, viz_dir):
    """Generate summary statistics and visualizations"""
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import Counter
    
    pred_counts = Counter(predictions)
    idx_to_label = {v: k for k, v in label_map.items()}
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    labels = [idx_to_label.get(i, f"Class_{i}") for i in sorted(pred_counts.keys())]
    counts = [pred_counts[i] for i in sorted(pred_counts.keys())]
    plt.bar(labels, counts, alpha=0.7, color='skyblue')
    plt.title('Prediction Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
 
    plt.subplot(1, 2, 2)
    plt.hist(confidences, bins=20, alpha=0.7, color='lightgreen')
    plt.title('Confidence Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.axvline(x=np.mean(confidences), color='red', linestyle='--', 
                label=f'Mean: {np.mean(confidences):.3f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/summary_statistics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    if attention_weights:
        layer_means = []
        cross_modal_attention = []
        
        for batch_attn in attention_weights:
            for layer_idx, layer_attn in enumerate(batch_attn):
                if layer_idx >= len(layer_means):
                    layer_means.append([])
                    cross_modal_attention.append([])
                
                layer_means[layer_idx].append(layer_attn.mean().item())
              
                img_to_text = layer_attn[:, :, 0, 1:].mean().item()
                cross_modal_attention[layer_idx].append(img_to_text)
    
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        layer_mean_avgs = [np.mean(layer_means[i]) for i in range(len(layer_means))]
        layer_mean_stds = [np.std(layer_means[i]) for i in range(len(layer_means))]
        plt.errorbar(range(len(layer_mean_avgs)), layer_mean_avgs, yerr=layer_mean_stds, 
                    marker='o', capsize=5)
        plt.title('Mean Attention by Layer')
        plt.xlabel('Layer')
        plt.ylabel('Mean Attention Weight')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        cross_modal_avgs = [np.mean(cross_modal_attention[i]) for i in range(len(cross_modal_attention))]
        cross_modal_stds = [np.std(cross_modal_attention[i]) for i in range(len(cross_modal_attention))]
        plt.errorbar(range(len(cross_modal_avgs)), cross_modal_avgs, yerr=cross_modal_stds, 
                    marker='s', capsize=5, color='orange')
        plt.title('Cross-Modal Attention by Layer')
        plt.xlabel('Layer')
        plt.ylabel('Image→Text Attention')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/attention_statistics.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Summary statistics:")
    print(f"Total predictions: {len(predictions)}")
    print(f"Mean confidence: {np.mean(confidences):.4f}")
    print(f"Std confidence: {np.std(confidences):.4f}")
    print(f"Class distribution: {dict(pred_counts)}")

if __name__ == "__main__":
    config_path = "config/config.yaml"
    checkpoint_path = "data/checkpoints/Sixth_visualization/model_epoch_4.pt"
    output_csv_path = "data/NEW_CSV/Prediction7_balanceclass_3000_with_viz.csv"
    device = "cuda"

    predict_and_visualize(config_path, checkpoint_path, output_csv_path, device, visualize_samples=True)