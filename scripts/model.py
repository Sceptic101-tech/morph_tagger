import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnablePositionalEncoding(nn.Module):
    '''Обучаемое позиционное кодирование'''
    def __init__(self, embed_dim:int, num_embed:int, padding_idx:int, batch_first:bool):
        super().__init__()
        self.batch_first = batch_first
        self.pos_embedings = nn.Embedding(num_embed, embed_dim, padding_idx, dtype=torch.float16)

    def forward(self, x:torch.Tensor):
        # x [B, N, D]
        seq_count = x.size(1) if self.batch_first else x.size(0)
        pos_idx = torch.arange(0, seq_count, dtype=torch.long).unsqueeze(0) # [1, N]
        pos_embed = self.pos_embedings(pos_idx) # [1, N, D]
        if not self.batch_first:
            pos_embed = pos_embed.permute(1, 0, 2) # [N, 1, D]
        return x + pos_embed
    
class MHAModel(nn.Module):
    pass