from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np


class InputConverter:
    def __init__(self, tokenizer=None, labelencoder=None, max_len=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labelencoder = labelencoder

    def _tokenizing(self, text_sequence):
        if not self.tokenizer:
            raise Exception("tokenizer is None")

        return self.tokenizer.texts_to_sequences(text_sequence)

    def _padding(self, encoding_token_list):
        return pad_sequences(encoding_token_list, maxlen=self.max_len, padding='post')

    def _encoding_label(self, label_sequence):
        if not self.labelencoder:
            raise Exception("labelencoder is None")

        return self.labelencoder.transfor(label_sequence)

    def main(self, text_sequence, label_sequence=None):
        encoded_token_list = self._tokenizing(text_sequence)
        input_x = self._padding(encoded_token_list)
        if not label_sequence:
            return input_x
        input_y = self._encoding_label(label_sequence)
        return input_x, input_y