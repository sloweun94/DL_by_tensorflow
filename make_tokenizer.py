from tensorflow.keras.preprocessing.text import Tokenizer
from soynlp.noun import LRNounExtractor_v2
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
import json

def make_tokenizer_from_keras(text_sequence, limit_word_counts=1, save_path=None):
    tf_tokenizer = Tokenizer(num_words=limit_word_counts, oov_token='OOV')
    tf_tokenizer.fit_on_texts(text_sequence)

    if save_path:
        with open(save_path, 'w', encoding='utf8') as file:
            file.write(json.dumps(tf_tokenizer.to_json(), ensure_ascii=False))

        print(f"Keras:Saving tokenizer (limit_count={limit_word_counts}) to: {save_path}")
    print(f"Keras: finished making tokenizer (limit_count={limit_word_counts})")
    return tf_tokenizer


def make_cohesion_score_from_soynlp(text_list, save_path=None):
    """
    word_extractor returns {word1:Scores(), word2:Scores()}
    :return
        {word1: cohesion_score, word2: cohesion_score}
    """
    word_extractor = WordExtractor(
        min_frequency=10,  # example
        min_cohesion_forward=0.05,
        min_right_branching_entropy=0.0
    )
    word_extractor.train(text_list)
    words = word_extractor.extract()
    score_dict = {word: score.cohesion_forward for word, score in words.items()}

    if save_path:
        with open(save_path, 'w', encoding='utf8') as file:
            file.write(json.dumps(score_dict, ensure_ascii=False))

        print(f"Soynlp:Saving cohesion's score to: {save_path}")
    print(f"Soynlp: finished making score")
    return score_dict


def make_noun_score_from_soynlp(text_list, save_path=None):
    """
    noun_extractor returns {word1:NounScore(frequency=000, score=1.0), word2:NounScore()}
    :return
        {word1: score, word2: score}
    """
    noun_extractor = LRNounExtractor_v2()
    nouns = noun_extractor.train_extract(text_list)  # list of str like

    noun_scores = {noun: score.score for noun, score in nouns.items()}

    if save_path:
        with open(save_path, 'w', encoding='utf8') as file:
            file.write(json.dumps(noun_scores, ensure_ascii=False))

        print(f"Soynlp:Saving noun's score to: {save_path}")
    print(f"Soynlp: finished making score")
    return noun_scores


def make_combine_score_from_soynlp(text_list, save_path=None):
    cohesion_score = make_cohesion_score_from_soynlp(text_list)
    noun_scores = make_noun_score_from_soynlp(text_list)

    score_dict = {noun: score + cohesion_score.get(noun, 0)
                  for noun, score in noun_scores.items()}
    score_dict.update(
        {subword: cohesion for subword, cohesion in cohesion_score.items()
         if not (subword in score_dict)}
    )

    if save_path:
        with open(save_path, 'w', encoding='utf8') as file:
            file.write(json.dumps(score_dict, ensure_ascii=False))

        print(f"Soynlp:Saving combine score [cohesion score + noun score] to: {save_path}")
    print(f"Soynlp: finished making score")
    return score_dict



if __name__ == '__main__':
    text_list = ['토크나이저 테스트 용도 입니다.']
    data_type = "tipa_sbjt_sk"

    score_type = "combine"
    save_path = f"{data_type}_{score_type}_score"
    score = make_combine_score_from_soynlp(text_list, save_path=save_path)
    tokenizer = LTokenizer(scores=score)


    limit_count = 2
    save_path = f"{data_type}_{limit_count}_tokenizer.json"
    tf_tokenizer = make_tokenizer_from_keras(text_list,limit_word_counts=2,save_path=save_path)
