import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, support_embeddings):
        # support_embeddings: (num_support, embed_dim)
        # Add batch dimension: (1, num_support, embed_dim)
        query = key = value = support_embeddings.unsqueeze(0)

        attn_out, attn_weights = self.mha(query, key, value)

        attn_out = attn_out.squeeze(0)  # (num_support, embed_dim)

        # Residual + LayerNorm
        out = self.norm(support_embeddings + self.dropout(attn_out))

        # attn_weights: (1, num_support, num_support) -> average over query dim
        per_sample_weights = attn_weights.mean(dim=1).squeeze(0)  # (num_support,)

        return out, per_sample_weights


class CrossAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_embedding, prototypes):
        # query_embedding: (num_query, embed_dim)
        # prototypes: (num_classes, embed_dim)

        query = query_embedding.unsqueeze(0)      # (1, num_query, embed_dim)
        key = value = prototypes.unsqueeze(0)     # (1, num_classes, embed_dim)

        attn_out, attn_weights = self.mha(query, key, value)

        attn_out = attn_out.squeeze(0)  # (num_query, embed_dim)

        # Residual + LayerNorm
        out = self.norm(query_embedding + self.dropout(attn_out))

        return out, attn_weights.squeeze(0)  # (num_query, num_classes)
