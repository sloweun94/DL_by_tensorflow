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
                tokenzier = json.load(file)
            self.input_converter = InputConverter(tokenizer=tokenzier, max_len=self.max_len)

        if label_encoder_path:
            with open(label_encoder_path, 'r', encoding='utf-8') as file:
                self.output_converter = json.load(file)

    def main(self, text_sequence):
        if isinstance(text_sequence, str):
            text_sequence = [text_sequence]
        input_data = self.input_converter.main(text_sequence=text_sequence)
        predicted_result = self.model.predict(input_data)
        predicted_result = np.argmax(predicted_result, axis=1)
        itc_sequence = [self.output_converter(idx) for idx in predicted_result]
        return itc_sequence


if __name__ == '__main__':
    import pandas as pd
    import random
    from sklearn.metrics import confusion_matrix, f1_score

    ds = pd.read_parquet("./data/filtering_basic_sk.parquet")
    choice_idx = random.sample(ds.index.to_list(), 1)
    sample_ds = ds.loc[ds.index.isin(choice_idx)]


    tokenizer_path = "tokenizer/tipa_sbjt_sk_2_tokenizer.json"

    model_path = './model/itc_subclass544_rnn_sk_220420'
    label_encoder_path = 'model/itc_subclass544_rnn_sk_220420/dataset/label_encoder.json'
    y_true = sample_ds.doc_subclass

    # model_path = './model/itc_section7_rnn_sk_220419'
    # label_encoder_path = 'model/itc_section7_rnn_sk_220419/dataset/label_encoder.json'
    # y_true = sample_ds.doc_subclass.str[:3]

    predictor = PredictingITC(model_path, tokenizer_path, label_encoder_path)
    result = predictor.main(sample_ds.text)

    confu = confusion_matrix(y_true,result)
    f1 = f1_score(y_true,result,average='weighted')
