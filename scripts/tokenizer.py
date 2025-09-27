import regex as re

class SeparatorTokenizer:
    def __init__(self):
        pass

    def tokenize(self, text:str, separator:str=None):
        """
        Разбивает строку на список токенов

        Args:
            text (str): исходный текст
            separator (str | None): символ/строка, по которой происходит
                разбиение. Если None – используется стандартное split() без
                аргументов (разделитель «пробел»)
        Returns:
            list[str]: список токенов
        """
        text = re.sub(r'[\w\s]+([^\w\s]+)', r' \1 ', text)  # Отделяем прбелом знаки препинания.
                                                            # Знаки препинания, следующие друг за другом, считаются одним токеном. Например: "...", "!?!"
        text = re.sub(r'[\t\n\r\f\v]', r' ', text)
        return text.split(sep=separator)
