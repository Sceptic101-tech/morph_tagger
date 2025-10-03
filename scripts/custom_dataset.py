class CustomDataset:
    def __init__(self, vectorizer, train_df, target_names:list[str], test_df=None, valid_df=None,):
        self._train_df = train_df
        self._test_df = test_df
        self._valid_df = valid_df
        self.vectorizer = vectorizer
        self.target_names = target_names
        self.set_dataframe_split('train')

    def set_dataframe_split(self, split:str):
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
        '''Возвращает словарь {source_x : source_vec(list[int]), target_names : target_vecs(list[int])}'''
        row = self.cw_df.iloc[index]
        src_vectorized, trg_vectorized = self.vectorizer.vectorize(row, self.target_names, use_vocab_max_len=True)
        vectorized = {key: value for key, value in trg_vectorized.items()}
        vectorized['source_x'] = src_vectorized
        return vectorized