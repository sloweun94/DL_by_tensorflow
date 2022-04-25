import os, re
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K

from convert import InputConverter

TOKENIZER_PATH = './tokenizer'
MODEL_PATH = './model'
DATA_PATH = './data'


class PredictingITC:
    def __init__(self, model_path, tokenizer_path, label_encoding_path):
        self.model = load_model(model_path)
        self.max_len = self.model.get_config()['layers'][0]['config']['batch_input_shape'][1]

        if tokenizer_path or label_encoding_path:
            self._load_converter(tokenizer_path, label_encoding_path)

    def _load_converter(self, tokenizer_path, label_encoder_path):
        if tokenizer_path:
            with open(tokenizer_path,'r',encoding='utf-8') as file:
                tokenzier_json = json.load(file)
            tokenzier = tokenizer_from_json(tokenzier_json)
            self.input_converter = InputConverter(tokenizer=tokenzier, max_len=self.max_len)

        if label_encoder_path:
            with open(label_encoder_path, 'r', encoding='utf-8') as file:
                self.output_converter_dict = json.load(file)

    def main(self, text_sequence, rank_limit=None):
        if isinstance(text_sequence, str):
            text_sequence = [text_sequence]
        input_data = self.input_converter.main(text_sequence=text_sequence)
        predicted_results = self.model.predict(input_data)

        if not rank_limit:
            predicted_result = np.argmax(predicted_results, axis=1)
            itc_sequence = [self.output_converter_dict[str(idx)] for idx in predicted_result]
            return itc_sequence

        predicted_label_percent_dict_list = []
        for predicted_ in predicted_results:

            # 확률 내림차순 정렬
            _rank_array = predicted_.argsort()[::-1]
            predicted_label_percent_dict = {}
            for rank_number in range(0, rank_limit):
                _predicted_label, _probability = _rank_array[rank_number], 100 * predicted_[_rank_array[rank_number]]
                _predicted_label = self.output_converter_dict[str(_predicted_label)]
                _probability = "{:2.2f}%".format(_probability)
                predicted_label_percent_dict.update({_predicted_label:_probability})
            predicted_label_percent_dict_list.append(predicted_label_percent_dict)
        return predicted_label_percent_dict_list


