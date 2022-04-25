
DATAPATH = 'data_folder'
#
#
# def show_length(sequence_data, xlim_rg=None, ylim_rg=None, plot_type='hist'):
#     sent_max_len = max(len(sample) for sample in sequence_data)
#     print('텍스트의 최대 길이 :{}'.format(sent_max_len))
#     print('텍스트의 평균 길이 :{}'.format(sum(map(len, sequence_data))/len(sequence_data)))
#     if plot_type == 'hist':
#         plt.hist([len(sample) for sample in sequence_data], bins=50)
#     if plot_type == 'box':
#         plt.figure(figsize=(12, 5))
#         plt.boxplot([len(sample) for sample in sequence_data], showmeans=True)
#     if xlim_rg:
#         plt.xlim(xlim_rg)
#     if ylim_rg:
#         plt.ylim(ylim_rg)
#
#     plt.xlabel('length of samples')
#     plt.ylabel('number of samples')
#     plt.show()
#
#
# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.layers import SimpleRNN, Embedding, Dense, LSTM, Dropout, Conv1D, GlobalMaxPooling1D
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# import matplotlib.pyplot as plt
# import datetime
# d_today = datetime.date.today()
#
# text_column_name = 'term'
# class_column_name = 'tecl_cd'
#
# parquet_path = './data_folder/target_tipa_terms_itc_20220325.parquet'
# origin_data_set = pd.read_parquet(parquet_path)
#
# data_index = origin_data_set.groupby(class_column_name).filter(lambda x: len(x)>=10)[class_column_name].index
# data = origin_data_set.iloc[data_index].reset_index(drop=True)
#
# # 데이터 분할
# data_X = data.term.tolist()
#
# class_depth = 'section'
#
# data_y = None
# if class_depth == 'section':
#     data_y = data[class_column_name].str[:3]
# elif class_depth == 'class':
#     data_y = data[class_column_name].str[:5]
# elif class_depth == 'sub':
#     data_y = data[class_column_name]
#
#
# X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, train_size=0.8, random_state=42,shuffle=True,stratify=data_y)
#
# # 정수 인코딩
# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
# label_encoder.fit(y_train)
# num_classes = len(label_encoder.classes_)
# y_train = label_encoder.transform(y_train)
# y_test = label_encoder.transform(y_test)
# idx_to_class = {idx:nm for idx, nm in enumerate(label_encoder.classes_[0])}
# class_to_idx = {nm:idx for idx, nm in enumerate(label_encoder.classes_[0])}
# print(num_classes)
#
# vocab_size=128091
# tokenizer = Tokenizer(num_words=vocab_size, oov_token='OOV')
# tokenizer.fit_on_texts(X_train)
# X_train = tokenizer.texts_to_sequences(X_train)
# X_test = tokenizer.texts_to_sequences(X_test)
#
# max_len = 300
#
# padded_X_train = pad_sequences(X_train, maxlen=max_len,padding="post")
# padded_X_test = pad_sequences(X_test, maxlen=max_len,padding="post")
#
# # 학습하기
# code_name = 'tipa'
# model_name = 'rnn'
#
# embedding_dim = 300
# hidden_units = max_len * 2
# dropout_ratio = 0.5
#
# with tf.device("/device:GPU:0"):
#     model = Sequential()
#     model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))
#     model.add(Dropout(dropout_ratio))
#     model.add(SimpleRNN(hidden_units))
#     model.add(Dropout(dropout_ratio))
#     model.add(Dense(num_classes, activation='softmax'))
#     print("{} 분류, {} model, {} 수준의 {}개 예측 model 학습".format(code_name, model_name, class_depth, num_classes))
#
#     checkpoint_path = f"./model/{code_name}_{class_depth}_{model_name}_{d_today}_best_model.h5"
#     checkpoint_dir = os.path.dirname(checkpoint_path)
#
#     if not os.path.isdir(checkpoint_dir):
#         os.makedirs(checkpoint_dir)
#
#     es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
#     mc = ModelCheckpoint(filepath=checkpoint_path, monitor='val_acc', mode='max', verbose=2, save_best_only=True)
#     # todo: add tensorboard
#     model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
#     history = model.fit(padded_X_train, y_train, batch_size=32, epochs=10, callbacks=[es, mc], validation_split=0.1, verbose=2)
#
#
#     if_test = False
#     if if_test:
#         model.evaluate(X_test, y_test)
#         model.predict()
#
#
# epochs = range(1, len(history.history['acc']) + 1)
# plt.plot(epochs, history.history['acc'])
# plt.plot(epochs, history.history['val_acc'])
# plt.title('model acc')
# plt.ylabel('acc')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()

from keras import backend as K


def recall(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_target_yn = K.round(K.clip(y_target, 0, 1))  # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))  # 예측값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn)

    # (True Positive + False Negative) = 실제 값이 1(Positive) 전체
    count_true_positive_false_negative = K.sum(y_target_yn)

    # Recall =  (True Positive) / (True Positive + False Negative)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())

    # return a single tensor value
    return recall


def precision(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))  # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
    y_target_yn = K.round(K.clip(y_target, 0, 1))  # 실제값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn)

    # (True Positive + False Positive) = 예측 값이 1(Positive) 전체
    count_true_positive_false_positive = K.sum(y_pred_yn)

    # Precision = (True Positive) / (True Positive + False Positive)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())

    # return a single tensor value
    return precision


def f1score(y_target, y_pred):
    _recall = recall(y_target, y_pred)
    _precision = precision(y_target, y_pred)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    _f1score = (2 * _recall * _precision) / (_recall + _precision + K.epsilon())

    # return a single tensor value
    return _f1score