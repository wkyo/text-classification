# coding: utf-8
from keras_preprocessing.text import Tokenizer
import jieba


def _load_stop_words(file):
    if isinstance(file, str):
        with open(file, 'rt', encoding='utf-8') as fp:
            words = fp.read().splitlines()
            return set(words)
    return None


class JiebaTokenizer(Tokenizer):
    def __init__(self, *args, stop_words=None, stop_words_path=None, jieba_dict_path=None, **kwargs):

        if stop_words_path:
            stop_words = _load_stop_words(stop_words_path)
        if isinstance(stop_words, list):
            stop_words = set(stop_words)
        self.stop_words = stop_words if stop_words is not None else set()

        super().__init__(*args, **kwargs)

    def __tokenizer(self, texts):
        """Use jieba tokenizer to segment text, blank or stop word will be ignored

        Returns:
            a generator with a list of words
        """
        for text in texts:
            text = text.strip()
            words = []
            for token in jieba.cut(text):
                token = token.strip()
                if token and token not in self.stop_words:
                    words.append(token)
            yield words

    def fit_on_texts(self, texts):
        texts = self.__tokenizer(texts)
        return super().fit_on_texts(texts)

    def texts_to_sequences_generator(self, texts):
        texts = self.__tokenizer(texts)
        return super().texts_to_sequences_generator(texts)

    def texts(self, sequences, sep=''):
        num_words = self.num_words
        oov_token_index = self.word_index.get(self.oov_token)

        _texts = []

        for seq in sequences:
            vect = []
            for num in seq:
                word = self.index_word.get(num)
                if word is not None:
                    if num_words and num >= num_words:
                        if oov_token_index is not None:
                            vect.append(self.index_word[oov_token_index])
                    else:
                        vect.append(word)
                elif self.oov_token is not None:
                    vect.append(self.index_word[oov_token_index])
            vect = sep.join(vect)
            _texts.append(vect)

        return _texts
