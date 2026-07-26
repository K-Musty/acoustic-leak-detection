import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = embed_dim ** -0.5
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, support_embeddings):
        Q = self.query_proj(support_embeddings)
        K = self.key_proj(support_embeddings)
        V = self.value_proj(support_embeddings)
        scores = torch.matmul(Q, K.T) * self.scale
        weights = F.softmax(scores, dim=-1)
        weighted = torch.matmul(weights, V)
        per_sample_weights = weights.mean(dim=0)
        return weighted, per_sample_weights

class CrossAttention(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = embed_dim ** -0.5
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, query_embedding, prototypes):
        Q = self.query_proj(query_embedding)
        K = self.key_proj(prototypes)
        V = self.value_proj(prototypes)
        scores = torch.matmul(Q, K.T) * self.scale
        weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(weights, V)
        return attended, weights
