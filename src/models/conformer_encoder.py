# src/models/conformer_encoder.py (Fixed)
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionModule(nn.Module):
    """Depthwise convolution module - FIXED."""
    def __init__(self, dim, kernel_size=31, dropout=0.1):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(dim, 2*dim, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            2*dim, 2*dim, kernel_size=kernel_size,
            padding=kernel_size//2, groups=2*dim
        )
        # FIX: GLU reduces 2*dim → dim, so pointwise_conv2 expects dim input
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, x):
        # x: (batch, seq, dim)
        x = x.transpose(1, 2)          # (batch, dim, seq)
        x = self.pointwise_conv1(x)    # (batch, 2*dim, seq)
        x = self.depthwise_conv(x)     # (batch, 2*dim, seq)
        x = F.glu(x, dim=1)            # (batch, dim, seq) ← GLU halves the channels
        x = self.pointwise_conv2(x)    # (batch, dim, seq) ← now expects dim, not 2*dim
        x = self.dropout(x)
        x = x.transpose(1, 2)          # (batch, seq, dim)
        return x

class ConformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, kernel_size=31, dropout=0.1):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(dim)

        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        self.conv = ConvolutionModule(dim, kernel_size, dropout)
        self.norm3 = nn.LayerNorm(dim)

        self.ffn2 = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm4 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (batch, seq, dim)
        x = x + 0.5 * self.ffn1(x)
        x = self.norm1(x)

        attn_out, _ = self.mha(x, x, x)
        x = x + attn_out
        x = self.norm2(x)

        x = x + self.conv(x)
        x = self.norm3(x)

        x = x + 0.5 * self.ffn2(x)
        x = self.norm4(x)
        return x

class ConformerEncoder(nn.Module):
    def __init__(self,
                 input_dim=2000,
                 output_dim=128,
                 num_heads=4,
                 ffn_dim=256,
                 num_layers=4,
                 kernel_size=31,
                 dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, ffn_dim)

        self.blocks = nn.ModuleList([
            ConformerBlock(ffn_dim, num_heads, ffn_dim, kernel_size, dropout)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(ffn_dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

    def forward(self, x):
        # x: (batch, input_dim)
        x = self.input_proj(x)
        x = x.unsqueeze(1)

        for block in self.blocks:
            x = block(x)

        x = x.squeeze(1)
        x = self.output_proj(x)
        x = self.norm_out(x)
        return x
