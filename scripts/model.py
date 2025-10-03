import math
import torch
import torch.nn as nn


class LearnablePositionalEncoding(nn.Module):
    '''Обучаемое позиционное кодирование'''
    def __init__(self, embed_dim:int, max_seq_len:int, padding_idx:int, batch_first:bool):
        super().__init__()
        self.batch_first = batch_first
        self.pos_embedings = nn.Embedding(max_seq_len, embed_dim, padding_idx)

    def forward(self, x:torch.Tensor):
        # x [B, N, D]
        if self.batch_first:
            seq_len = x.size(1)
            pos_idx = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, S]
        else:
            seq_len = x.size(0)
            pos_idx = torch.arange(seq_len, device=x.device).unsqueeze(1)  # [S, 1]
        
        pos_embed = self.pos_embedings(pos_idx) # [1, S, D] | [S, 1, D]
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

        self.encoder_ff =  nn.Sequential(nn.Linear(attention_dim, dim_encoder_ff, bias), nn.ReLU(), nn.Dropout(dropout),\
                                         nn.Linear(dim_encoder_ff, attention_dim, bias), nn.Dropout(dropout))
    
    def forward(self, x, key_padding_mask):
        x_norm = self.norm1(x)
        query, key, value = (self.query_ff(x_norm), self.key_ff(x_norm), self.value_ff(x_norm))

        attention_out, attention_out_weights = self.attention(query, key, value, key_padding_mask=key_padding_mask)
        x = x + self.dropout1(attention_out) # residual
        encoder_out = self.encoder_ff(self.norm2(x))
        return (x + encoder_out)


class MHAModel(nn.Module):
    def __init__(self, max_seq_len:int, num_embeddings:int, embedding_dim:int, attention_dim:int, num_heads:int, num_layers:int, dim_classifier_ff_hidden:int, dim_encoder_ff:int,\
                 classifiers_names_params:dict[str, int], dropout:float, temperature:float, batch_first:bool, bias:bool=True, padding_idx:int=0):
        # classifiers_names_params: ожидается словарь, где ключ - название признака, а значение - размерность словаря признака
        super().__init__()

        self.batch_first = batch_first
        self.padding_idx = padding_idx
        self.num_layers = num_layers
        self.temperature = temperature
        self.classifiers_names_params = classifiers_names_params

        self.embedings = nn.Embedding(num_embeddings, embedding_dim, padding_idx)
        self.positional_encoding = LearnablePositionalEncoding(embedding_dim, max_seq_len, padding_idx, batch_first)
        self.embed_to_encod_proj = nn.Linear(embedding_dim, attention_dim, bias)
        self.layer_norm = nn.LayerNorm(attention_dim)
        self.encoder_stack = nn.ModuleList([EncoderBlock(attention_dim, num_heads, dropout, dim_encoder_ff, bias, batch_first) for _ in range(num_layers)])

        self.final_classifiers = nn.ModuleDict({key:nn.Sequential(
            nn.Linear(attention_dim, dim_classifier_ff_hidden, bias), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_classifier_ff_hidden, value, bias))\
                for key, value in classifiers_names_params.items()})


    def forward(self, x, apply_softmax:bool=False)->dict[str:torch.Tensor]:
        # x [B, S]
        key_padding_mask = (x == self.padding_idx)
        embedded = self.embedings(x) # [B, S, E]
        embedded = self.positional_encoding(embedded)
        x = self.embed_to_encod_proj(embedded) # [B, S, D]
        for layer in range(self.num_layers):
            x = self.encoder_stack[layer](x, key_padding_mask)
        # x [B, S, D]
        x = self.layer_norm(x)
        logits = {}
        for key, value in self.classifiers_names_params.items():
            logits[key] = self.final_classifiers[key](x) # [B, S, num_classes_key]

        if apply_softmax:
            for key, _ in logits.items():
                logits[key] = nn.functional.softmax(logits[key]/self.temperature, dim=-1)
        return logits
