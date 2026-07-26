import torch
import torch.nn as nn

class ConformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout)
        )
        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn2 = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
    
    def forward(self, x):
        # x: (batch, 1, dim)
        x = x + self.ffn1(x)
        x = self.norm1(x)
        attn_out, _ = self.mha(x, x, x)
        x = x + attn_out
        x = self.norm2(x)
        x = x + self.ffn2(x)
        x = self.norm3(x)
        return x

class ConformerEncoder(nn.Module):
    def __init__(self, 
                 input_dim=2000,
                 output_dim=128,
                 num_heads=4,
                 ffn_dim=256,
                 num_layers=4,
                 dropout=0.1):
        super().__init__()
        
        # Project raw waveform to ffn_dim
        self.input_proj = nn.Linear(input_dim, ffn_dim)
        
        # Conformer blocks
        self.blocks = nn.ModuleList([
            ConformerBlock(ffn_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(ffn_dim, output_dim)
    
    def forward(self, x):
        # x: (batch, input_dim)
        x = self.input_proj(x)      # (batch, ffn_dim)
        x = x.unsqueeze(1)          # (batch, 1, ffn_dim)
        
        for block in self.blocks:
            x = block(x)            # (batch, 1, ffn_dim)
        
        x = x.squeeze(1)            # (batch, ffn_dim)
        x = self.output_proj(x)     # (batch, output_dim)
        return x
