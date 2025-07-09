import torch
import torch.nn as nn

class CrossModalFusion(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, inputs):  # list of [B, D] tensors
        stacked = torch.stack(inputs, dim=1)  # [B, M, D]
        fused, _ = self.attn(stacked, stacked, stacked)  # self-attention across modalities
        return self.proj(fused.mean(dim=1))  # [B, D]
