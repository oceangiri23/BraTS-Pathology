import torch
import torch.nn as nn

from models.custom_transformer_encoder_layer import CustomTransformerEncoderLayer
from models.custom_cross_attention_layer import CustomCrossAttentionLayer

class modifiedCrossAttentionTransformer(nn.Module):
    def __init__(self, image_embedding_dim, text_embedding_dim,
                 hidden_dim, num_heads, num_layers, num_classes, dropout):
        super().__init__()
        self.image_proj = nn.Linear(image_embedding_dim, hidden_dim)
        self.text_proj = nn.Linear(text_embedding_dim, hidden_dim)

        self.layers = nn.ModuleList([
            CustomCrossAttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, image_embeds, text_embeds, attention_mask=None, return_attn=False):
        B, seq_len, _ = text_embeds.shape

        img_proj = self.image_proj(image_embeds)  # [B, L_img, D]
        if img_proj.dim() == 2:
            img_proj = img_proj.unsqueeze(1)
        txt_proj = self.text_proj(text_embeds)    # [B, L_text, D]

        padding_mask = ~attention_mask.bool() if attention_mask is not None else None

        attn_weights_all = []
        output = txt_proj
        for layer in self.layers:
            output, attn_weights = layer(output, img_proj, key_padding_mask=None)
            if return_attn:
                attn_weights_all.append(attn_weights)

        # Pool (mean) or CLS-like (first token) – depending on setup
        pooled = output[:, 0]  # or output.mean(dim=1)

        logits = self.classifier(pooled)

        if return_attn:
            return logits, attn_weights_all
        else:
            return logits