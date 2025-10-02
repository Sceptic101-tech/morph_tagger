import torch
import numpy as np
import pandas as pd
from scripts.vocabulary import Vocabulary

class Vectorizer:
    def __init__(self, src_vocab:Vocabulary, trg_vocabs:dict[Vocabulary], max_src_len:int, mask_idx:int):
        self.max_src_len = max_src_len
        self.src_vocab = src_vocab
        self.target_cnt = len(trg_vocabs)
        self.trg_vocabs = trg_vocabs
        self.mask_idx = mask_idx


    def get_indices(self, tokenized_text:list[str], cw_vocab:Vocabulary, add_bos:bool=True, add_eos:bool=True)->list[int]:
        indices = []
        if add_bos:
            indices.append(cw_vocab.bos_idx)
        for token in tokenized_text:
            indices.append(cw_vocab.token_to_idx[token])
        if add_eos:
            indices.append(cw_vocab.eos_idx)
        return indices
    
    
    def _vectorize(self, indices:list[int], use_vocab_max_len:bool):
        # Для обучения используем максимальную длину предложения, для инференса - длину текущего предложения
        if use_vocab_max_len:
            seq_len = self.max_src_len
        else:
            seq_len = len(indices)
        
        padded = [self.mask_idx] * seq_len # Заполнение индексом маскировочного токена
        padded[:len(indices)] = indices # Заполнение индексами реальных токенов
        return padded


    def vectorize(self, df_row:pd.Series, target_names:list[str], use_vocab_max_len:bool=True)->tuple[list[int], dict[str, list[int]]]:
        '''Принимает строку датафрейма, возвращает множество\n
        (векторизованный source, {название target : векторизованный target})\n
        Такая структура необходима для векторизации нескольких целевых меток одновременно.'''
        src_indices = self.get_indices(df_row['source_tokens'], self.src_vocab)
        src_vectorized = self._vectorize(src_indices, use_vocab_max_len)
        trg_vectorized = {}
        for target_name in target_names:
            trg_indices = self.get_indices(df_row[target_name], self.trg_vocabs[target_name], add_bos=False, add_eos=False)
            trg_vectorized[target_name] = self._vectorize(trg_indices, use_vocab_max_len)
        
        return (src_vectorized, trg_vectorized)
        