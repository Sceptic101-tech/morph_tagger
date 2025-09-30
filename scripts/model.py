import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnablePositionalEncoding(nn.Module):
    '''Обучаемое позиционное кодирование'''
    def __init__(self, embed_dim:int, max_seq_len:int, padding_idx:int, batch_first:bool):
        super().__init__()
        self.batch_first = batch_first
        self.pos_embedings = nn.Embedding(max_seq_len, embed_dim, padding_idx)

    def forward(self, x:torch.Tensor):
        # x [B, N, D]
        seq_count = x.size(1) if self.batch_first else x.size(0)
        pos_idx = torch.arange(0, seq_count, dtype=torch.long, device=x.device).unsqueeze(0) # [1, N]
        pos_embed = self.pos_embedings(pos_idx) # [1, N, D]
        if not self.batch_first:
            pos_embed = pos_embed.permute(1, 0, 2) # [N, 1, D]
        return x + pos_embed
    
class EncoderBlock(nn.Module):
    '''Блок энкодера с многоловым вниманием и полносвязной сетью'''
    def __init__(self, attention_dim, num_heads, dropout, dim_encoder_ff, bias:bool=True, batch_first:bool=True):
        super().__init__()
        self.query_ff = nn.Linear(attention_dim, attention_dim, bias)
        self.key_ff = nn.Linear(attention_dim, attention_dim, bias)
        self.value_ff = nn.Linear(attention_dim, attention_dim, bias)

        self.norm1 = nn.LayerNorm(attention_dim)
        self.attention = nn.MultiheadAttention(attention_dim, num_heads, dropout, bias=bias, batch_first=batch_first)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(attention_dim)

        self.encoder_ff =  nn.Sequential(nn.Linear(attention_dim, dim_encoder_ff, bias), nn.GELU(), nn.Dropout(dropout),\
                                         nn.Linear(dim_encoder_ff, attention_dim, bias), nn.Dropout(dropout))
        self.norm3 = nn.LayerNorm(attention_dim)
    
    def forward(self, x, key_padding_mask):
        x_norm = self.norm1(x)
        query, key, value = self.query_ff(x_norm), self.key_ff(x_norm), self.value_ff(x_norm)

        attention_out, attention_out_weights = self.attention(query, key, value, key_padding_mask=key_padding_mask)
        x = x + self.dropout1(attention_out) # residual
        encoder_out = self.encoder_ff(self.norm2(x))
        return self.norm3(x + encoder_out)
    
# На данный момент используется только одна ff сеть для классификации ОДНОГО признака для каждого токена

class MHAModel(nn.Module):
    def __init__(self, max_seq_len:int, num_embeddings:int, embedding_dim:int, attention_dim:int, num_heads:int, num_layers:int, dim_classifier_ff_hidden:int, dim_encoder_ff:int, num_cls:int,\
                    dropout:float, temperature:float, batch_first:bool, bias:bool=True, padding_idx:int=0):
        super().__init__()

        self.batch_first = batch_first
        self.padding_idx = padding_idx
        self.num_layers = num_layers
        self.temperature = temperature

        self.embedings = nn.Embedding(num_embeddings, embedding_dim, padding_idx)
        self.positional_encoding = LearnablePositionalEncoding(embedding_dim, max_seq_len, padding_idx, batch_first)
        self.embed_to_encod_proj = nn.Linear(embedding_dim, attention_dim, bias)
        self.layer_norm = nn.LayerNorm(attention_dim)
        self.encoder_stack = nn.ModuleList([EncoderBlock(attention_dim, num_heads, dropout, dim_encoder_ff, bias, batch_first) for _ in range(num_layers)])

        self.classifier = nn.Sequential(
            nn.Linear(attention_dim, dim_classifier_ff_hidden, bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_classifier_ff_hidden, num_cls, bias))

    def forward(self, x, apply_softmax:bool=True):
        # x [B, S]
        key_padding_mask = x == self.padding_idx
        embedded = self.embedings(x) # [B, S, E]
        embedded = self.positional_encoding(embedded)
        x = self.embed_to_encod_proj(embedded) # [B, S, D]
        x = self.layer_norm(x)
        for layer in range(self.num_layers):
            x = self.encoder_stack[layer](x, key_padding_mask)

        logits = self.classifier(x) # [B, S, num_classes]
        if apply_softmax:
            logits = nn.functional.softmax(logits/self.temperature, dim=-1)
        return logits
