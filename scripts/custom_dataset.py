import torch

class CustomDataset:
    def __init__(self, vectorizer, train_df, target_names:list[str], add_bos_eos_tokens:bool=True, test_df=None, valid_df=None):
        """
        Создаёт Dataset для обучения/инференса.

        Parameters
        ----------
        vectorizer : Vectorizer
            Объект, преобразующий токены в индексы.
        train_df : pandas.DataFrame
            DataFrame с обучающим набором данных.
        target_names : list[str]
            Список названий целевых меток.
        test_df : pandas.DataFrame, optional
            DataFrame с тестовым набором данных.
        valid_df : pandas.DataFrame, optional
            DataFrame с валидационным набором данных.
        """
        self._train_df = train_df
        self._test_df = test_df
        self._valid_df = valid_df
        self.vectorizer = vectorizer
        self.target_names = target_names
        self.add_bos_eos_tokens = add_bos_eos_tokens
        self.set_dataframe_split('train')

    def set_dataframe_split(self, split:str):
        """
        Устанавливает текущий DataFrame для работы.

        Parameters
        ----------
        split : str
            Один из {'train', 'test', 'validation'}.
        """
        match split:
            case 'train':
                self.cw_df = self._train_df
            case 'test':
                self.cw_df = self._test_df
            case 'validation':
                self.cw_df = self._valid_df
            case _:
                raise ValueError('Неверное значение параметра split. Допустимые значения: train, test, validation')

    def __len__(self):
        return len(self.cw_df)
    
    def __getitem__(self, index:int):
        '''Возвращает словарь {source_x : source_vec(tensor), target_names : target_vecs(tensor)}'''
        row = self.cw_df.iloc[index]
        src_vectorized, trg_vectorized = self.vectorizer.vectorize(row, self.target_names, use_vocab_max_len=True, add_bos_eos_tokens=self.add_bos_eos_tokens)
        vectorized = {}
        for key, value in trg_vectorized.items():
            vectorized[key] = torch.tensor(value)  # Преобразуем в тензор
        vectorized['source_x'] = torch.tensor(src_vectorized)  # Преобразуем в тензор
        return vectorized