import json

class Vocabulary:
    """
    Класс для работы со словарём токен‑индекс
    
    Внутри хранит два отображения: token->idx и idx->token, а также специальные токены (MASK, UNK, BOS, EOS)
    """
    def __init__(self, token_to_idx:dict=None, bos_token:str='<BOS>', eos_token:str='<EOS>', mask_token:str='<MASK>', unk_token='<UNK>', add_bos_eos_tokens:bool=True):
        self.add_bos_eos_tokens = add_bos_eos_tokens
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.mask_token = mask_token
        self.unk_token = unk_token

        if token_to_idx is not None:
            self.token_to_idx = token_to_idx
            self.idx_to_token = {value: key for key, value in token_to_idx.items()}
        else:
            self.token_to_idx = {}
            self.idx_to_token = {}
            self.mask_idx = self.add_token(mask_token)
            self.unk_idx = self.add_token(unk_token)
            if add_bos_eos_tokens:
                self.bos_idx = self.add_token(bos_token)
                self.eos_idx = self.add_token(eos_token)

    def add_token(self, token:str)->int:
        """Добавляет токен в словарь. Возвращает индекс токена"""
        if token not in self.token_to_idx:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token
            return idx
        else:
            return self.token_to_idx[token]
        
    def add_tokens(self, tokens:list[str])->list[int]:
        return [self.add_token(token) for token in tokens]
        
    def to_serializable(self) -> dict:
        """
        Возвращает словарь, пригодный для сериализации (JSON)
        """
        return {
            'token_to_idx': self.token_to_idx,
            'bos_token': self.bos_token,
            'eos_token': self.eos_token,
            'mask_token': self.mask_token,
            'unk_token': self.unk_token,
            'add_auxiliary_tokens': self.add_bos_eos_tokens}

    @classmethod
    def from_serializable(cls, serializable: dict):
        """Создаёт объект Vocabulary из сериализованного представления"""
        return cls(**serializable)

    def to_json(self, filepath: str):
        """Сохраняет словарь в JSON‑файл"""
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(self.to_serializable(), file, ensure_ascii=False)

    @classmethod
    def from_json(cls, filepath: str):
        """Загружает словарь из JSON‑файла"""
        with open(filepath, 'r', encoding='utf-8') as file:
            return cls.from_serializable(json.load(file))
        
    def __len__(self)->int:
        """Возвращает длину словаря"""
        return len(self.token_to_idx)
    

