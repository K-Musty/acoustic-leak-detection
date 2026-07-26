import torch
import torch.nn as nn
from models.conformer_encoder import ConformerEncoder
from models.attention import SelfAttention, CrossAttention

class PrototypicalNetwork(nn.Module):
    def __init__(self, 
                 input_dim=2000,
                 embed_dim=128,
                 use_self_attention=True,
                 use_cross_attention=True,
                 conformer_kwargs=None):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.use_self_attention = use_self_attention
        self.use_cross_attention = use_cross_attention
        
        if conformer_kwargs is None:
            conformer_kwargs = {}
        
        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            output_dim=embed_dim,
            **conformer_kwargs
        )
        
        if use_self_attention:
            self.self_attn = SelfAttention(embed_dim)
        if use_cross_attention:
            self.cross_attn = CrossAttention(embed_dim)
    
    def forward(self, support_X, support_y, query_X):
        support_emb = self.encoder(support_X)
        query_emb = self.encoder(query_X)
        
        # Compute prototypes
        prototypes = []
        for cls in torch.unique(support_y):
            mask = support_y == cls
            proto = support_emb[mask].mean(dim=0)
            prototypes.append(proto)
        prototypes = torch.stack(prototypes)
        
        # Self-attention (optional)
        if self.use_self_attention:
            support_emb, _ = self.self_attn(support_emb)
            prototypes = []
            for cls in torch.unique(support_y):
                mask = support_y == cls
                proto = support_emb[mask].mean(dim=0)
                prototypes.append(proto)
            prototypes = torch.stack(prototypes)
        
        # Cross-attention (optional)
        if self.use_cross_attention:
            query_emb, _ = self.cross_attn(query_emb, prototypes)
        
        distances = torch.cdist(query_emb, prototypes)
        return distances
    
    def get_embeddings(self, X):
        return self.encoder(X)
