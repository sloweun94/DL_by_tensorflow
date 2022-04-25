import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from model import InputConverter


with open("./tokenizer/tipa_sbjt_sk_1_tokenizer.json") as file:
    json_dict = json.load(file)
tf_tokenizer = tokenizer_from_json(json_dict)
vocab_size = tf_tokenizer.num_words
index_to_word = tf_tokenizer.index_word

input_converter = InputConverter(tokenizer=tf_tokenizer, labelencoder=label_encoder, max_len=max_len)
input_X_train, input_y_train = input_converter.main(X_data, y_data)

if save_path:
    with open(f'{save_path}/label_encoder.json', 'w', encoding='utf8') as file:
        file.write(json.dumps(idx_to_class, ensure_ascii=False))