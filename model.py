import os, re, json
import pandas as pd
import numpy as np
import datetime
d_today = datetime.date.today()

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# todo: f1score - sklearn 사용 코드 작성하기 - 코드 따로 분리


def ready_dataset(X_data, y_data, tokenizer, label_encoder, max_len):
    input_converter = InputConverter(tokenizer=tokenizer, labelencoder=label_encoder, max_len=max_len)
    input_X_train, input_y_train = input_converter.main(X_data, y_data)

    return input_X_train, input_y_train





def train_model(X_train, y_train, tokenizer, token_max_len, model, model_save_path):
    # encoding label
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    num_classes = len(label_encoder.classes_)
    input_y_train = label_encoder.transform(y_train)
    idx_to_class = {idx: nm for idx, nm in enumerate(label_encoder.classes_)}

    with open(f'{model_save_path}/label_encoder.json', 'w', encoding='utf8') as file:
        file.write(json.dumps(idx_to_class, ensure_ascii=False))

    # tokenizing text
    input_converter = InputConverter(tokenizer=tokenizer)
    input_X_train = input_converter.main(X_train, max_len=token_max_len, padding='post')


    # todo: model input방법 고민하기


    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
    mc = ModelCheckpoint(filepath=f"{model_save_path}/checkpoint", monitor='val_acc', mode='max', verbose=2, save_best_only=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    history = model.fit(input_X_train, input_y_train, batch_size=32, epochs=10, callbacks=[es, mc], validation_split=0.1, verbose=2)