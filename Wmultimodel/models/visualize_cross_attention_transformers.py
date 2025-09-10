import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

class modifiedCrossAttentionTransformer(nn.Module):
    def __init__(self,
                 image_embedding_dim: int = 2048,
                 text_embedding_dim: int = 1024,
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 num_classes: int = 9,
                 dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        self.image_proj = nn.Linear(image_embedding_dim, hidden_dim)
        self.text_proj = nn.Linear(text_embedding_dim, hidden_dim)

        
        self.attention_weights = []
        
        
        self.transformer_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dropout=dropout,
                batch_first=True,
            ) for _ in range(num_layers)
        ])

       
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, image_embeds, text_embeds, attention_mask=None, return_attention=False):
        """
        image_embeds: [B, image_embedding_dim]
        text_embeds: [B, seq_len, text_embedding_dim]
        attention_mask: [B, seq_len] -> used to create padding mask
        return_attention: bool -> whether to return attention weights
        """
        B, seq_len, _ = text_embeds.shape

        img_proj = self.image_proj(image_embeds).unsqueeze(1)      # [B, 1, hidden_dim]
        txt_proj = self.text_proj(text_embeds)                     # [B, seq_len, hidden_dim]

        combined = torch.cat([img_proj, txt_proj], dim=1)

        if attention_mask is not None:
            padding_mask = ~attention_mask.bool()
            padding_mask = torch.cat([
                torch.zeros((B, 1), dtype=torch.bool, device=padding_mask.device),
                padding_mask
            ], dim=1)  # [B, 1 + seq_len]
        else:
            padding_mask = None

        self.attention_weights = []

        x = combined
        for layer in self.transformer_layers:
            x, attn_weights = layer(x, src_key_padding_mask=padding_mask, return_attention=True)
            if return_attention:
                self.attention_weights.append(attn_weights)

        cls_token = x[:, 0]  # [B, hidden_dim]

        logits = self.classifier(cls_token)  # [B, num_classes]
        
        if return_attention:
            return logits, self.attention_weights, {
                'image_embeds': img_proj,
                'text_embeds': txt_proj,
                'combined_embeds': combined,
                'final_embeds': x
            }
        
        return logits


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, batch_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_key_padding_mask=None, return_attention=False):
 
        attn_output, attn_weights = self.self_attn(
            src, src, src, 
            key_padding_mask=src_key_padding_mask,
            need_weights=return_attention
        )
        
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
 
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        if return_attention:
            return src, attn_weights
        return src


class AttentionVisualizer:
    def __init__(self, model, tokenizer, label_map):
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.idx_to_label = {v: k for k, v in label_map.items()}
    
    def visualize_attention_heatmap(self, 
                                   image_embeds, 
                                   text_embeds, 
                                   attention_mask, 
                                   input_ids,
                                   true_label=None,
                                   save_path=None,
                                   layer_idx=0,
                                   head_idx=0):
        """
        Visualize attention weights as heatmap
        """
        self.model.eval()
        with torch.no_grad():
            logits, attention_weights, embeddings = self.model(
                image_embeds, text_embeds, attention_mask, return_attention=True
            )
            
        pred_label = torch.argmax(logits, dim=1).item()
        
        attn = attention_weights[layer_idx][0, head_idx].cpu().numpy()  # [seq_len, seq_len]
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        tokens = ['[IMG]'] + tokens[:len(tokens)]  # Add image token
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        mask = attention_mask[0].cpu().numpy()
        valid_len = mask.sum() + 1  # +1 for image token
        
        im1 = ax1.imshow(attn[:valid_len, :valid_len], cmap='Blues', aspect='auto')
        ax1.set_xticks(range(valid_len))
        ax1.set_yticks(range(valid_len))
        ax1.set_xticklabels(tokens[:valid_len], rotation=45, ha='right')
        ax1.set_yticklabels(tokens[:valid_len])
        ax1.set_title(f'Full Attention Matrix (Layer {layer_idx}, Head {head_idx})')
        plt.colorbar(im1, ax=ax1)
        
        img_to_text_attn = attn[0, 1:valid_len]  # Image token attending to text tokens
        text_to_img_attn = attn[1:valid_len, 0]  # Text tokens attending to image token
        
        ax2.bar(range(len(img_to_text_attn)), img_to_text_attn, alpha=0.7, label='Image→Text', color='blue')
        ax2.bar(range(len(text_to_img_attn)), text_to_img_attn, alpha=0.7, label='Text→Image', color='red')
        ax2.set_xticks(range(len(tokens[1:valid_len])))
        ax2.set_xticklabels(tokens[1:valid_len], rotation=45, ha='right')
        ax2.set_ylabel('Attention Weight')
        ax2.set_title('Cross-Modal Attention Weights')
        ax2.legend()
        
        pred_class = self.idx_to_label.get(pred_label, f"Class_{pred_label}")
        true_class = self.idx_to_label.get(true_label, f"Class_{true_label}") if true_label is not None else "Unknown"
        fig.suptitle(f'Predicted: {pred_class} | True: {true_class}', fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_embedding_similarity(self, 
                                     image_embeds, 
                                     text_embeds, 
                                     attention_mask,
                                     save_path=None):
        """
        Visualize similarity between image and text embeddings
        """
        self.model.eval()
        with torch.no_grad():
            logits, attention_weights, embeddings = self.model(
                image_embeds, text_embeds, attention_mask, return_attention=True
            )
       
        img_embed = embeddings['image_embeds'][0, 0].cpu().numpy()  # [hidden_dim]
        txt_embeds = embeddings['text_embeds'][0].cpu().numpy()     # [seq_len, hidden_dim]
        
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([img_embed], txt_embeds)[0]
        
        plt.figure(figsize=(12, 6))
        
        mask = attention_mask[0].cpu().numpy()
        valid_similarities = similarities[:mask.sum()]
        
        plt.bar(range(len(valid_similarities)), valid_similarities, color='green', alpha=0.7)
        plt.xlabel('Text Token Position')
        plt.ylabel('Cosine Similarity with Image')
        plt.title('Image-Text Embedding Similarity')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_attention_rollout(self, 
                                   image_embeds, 
                                   text_embeds, 
                                   attention_mask,
                                   input_ids,
                                   save_path=None):
        """
        Visualize attention rollout across all layers
        """
        self.model.eval()
        with torch.no_grad():
            logits, attention_weights, embeddings = self.model(
                image_embeds, text_embeds, attention_mask, return_attention=True
            )
        rollout = self._compute_rollout(attention_weights, attention_mask)
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        tokens = ['[IMG]'] + tokens
        
        mask = attention_mask[0].cpu().numpy()
        valid_len = mask.sum() + 1
        
        plt.figure(figsize=(12, 10))
        
        im = plt.imshow(rollout[:valid_len, :valid_len], cmap='Reds', aspect='auto')
        plt.xticks(range(valid_len), tokens[:valid_len], rotation=45, ha='right')
        plt.yticks(range(valid_len), tokens[:valid_len])
        plt.title('Attention Rollout Across All Layers')
        plt.colorbar(im)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _compute_rollout(self, attention_weights, attention_mask):
        """
        Compute attention rollout across layers
        """
        layer_attentions = []
        for layer_attn in attention_weights:
            avg_attn = layer_attn[0].mean(dim=0).cpu().numpy()  # Average across heads
            layer_attentions.append(avg_attn)
        
        rollout = layer_attentions[0]
        for i in range(1, len(layer_attentions)):
            rollout = np.matmul(rollout, layer_attentions[i])
        
        return rollout
    
    def visualize_layer_comparison(self, 
                                  image_embeds, 
                                  text_embeds, 
                                  attention_mask,
                                  input_ids,
                                  save_path=None):
        """
        Compare attention patterns across different layers
        """
        self.model.eval()
        with torch.no_grad():
            logits, attention_weights, embeddings = self.model(
                image_embeds, text_embeds, attention_mask, return_attention=True
            )
        
        num_layers = len(attention_weights)
        fig, axes = plt.subplots(1, num_layers, figsize=(6*num_layers, 5))
        
        if num_layers == 1:
            axes = [axes]
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        tokens = ['[IMG]'] + tokens
        
        mask = attention_mask[0].cpu().numpy()
        valid_len = mask.sum() + 1
        
        for layer_idx, ax in enumerate(axes):
            attn = attention_weights[layer_idx][0].mean(dim=0).cpu().numpy()
            
            im = ax.imshow(attn[:valid_len, :valid_len], cmap='Blues', aspect='auto')
            ax.set_title(f'Layer {layer_idx}')
            ax.set_xticks(range(valid_len))
            ax.set_yticks(range(valid_len))
            if layer_idx == 0:
                ax.set_yticklabels(tokens[:valid_len])
            ax.set_xticklabels(tokens[:valid_len], rotation=45, ha='right')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()