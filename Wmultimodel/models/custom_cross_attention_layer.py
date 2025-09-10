
import torch
import torch.nn as nn

class CustomCrossAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_proj, image_proj, key_padding_mask=None, return_reverse_attn=False):
        
        attn_output, attn_weights = self.cross_attn(
            query=text_proj,
            key=image_proj,
            value=image_proj,
            key_padding_mask=None,
            need_weights=True,
            average_attn_weights=False
        )
        if return_reverse_attn:
        
            _, reverse_attn_weights = self.cross_attn(
            query=image_proj, key=text_proj, value=text_proj,
            need_weights=True, average_attn_weights=False
        )
            return out, attn_weights, reverse_attn_weights
        
        out = self.norm(text_proj + self.dropout(attn_output))
        return out, attn_weights
