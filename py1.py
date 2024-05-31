# processing the emotion model
# Here is the code that use to import all needed library
"""
    The code imports the necessary libraries, preprocesses the data, trains a model, and defines a
    function `mot(text)` that takes a sentence as input and predicts the emotion associated with it.

    :param text: The "text" parameter is the input text for which you want to predict the emotion
    :return: The code is returning the predicted emotion label for a given input text.
"""
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import pandas as pd
import numpy as np
import tensorflow as tf
from model_processing_1 import model_processing

# This is the constant indicated in this code.
vocab_size = 20000
embedding_dim = 100
max_length = 300
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

# import data
"""
    The code provided is for training a model to classify emotions in text data and using the trained
    model to predict the emotion of a given sentence.

    :param text: The "text" parameter is the input sentence for which you want to predict the emotion
    :return: The function `mot(text)` returns the predicted emotion label for the given input text.
    """
d1 = pd.read_csv("archive2/vn/train.txt", delimiter=';', names=['sentence', 'label'], header=None)

train_df = d1.dropna()

test_df = pd.read_csv("archive2/vn/test.txt", delimiter=';', names=['sentence', 'label'], header=None)
test_df = test_df.dropna()

emotion = train_df['label'].unique()
print(emotion)
emotion_dict = {}
emotion_dict = {v: i for i, v in enumerate(emotion)}

train_df['label'] = train_df['label'].apply(lambda x: emotion_dict.get(x))
test_df['label'] = test_df['label'].apply(lambda x: emotion_dict.get(x))

train_sentence = [sentence for sentence in train_df['sentence']]
test_sentence = [sentence for sentence in test_df['sentence']]

tokenizer = Tokenizer(oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentence)
tokenizer.word_index

train_sequence = tokenizer.texts_to_sequences(train_sentence)
train_padded = pad_sequences(train_sequence, padding='post',
                             maxlen=max_length,
                             truncating=trunc_type)

test_sequence = tokenizer.texts_to_sequences(test_sentence)
test_padded = pad_sequences(test_sequence, padding='post', maxlen=max_length,
                            truncating=trunc_type)



Model_processing = model_processing(train_sentence=train_sentence,
                                    test_sentence=test_sentence)

Model_processing.tokenizer_def()

train_label = []
for i in train_df['label']:
    tmp = np.zeros(6)
    tmp[int(i)] = 1
    train_label.append(tmp)
train_label = np.array(train_label)
test_label = []
for i in test_df['label']:
    try:
        tmp = np.zeros(6)
        tmp[int(i)] = 1
        test_label.append(tmp)
    except:
        continue
#----------EN--------#
d2 = pd.read_csv("archive2/train.txt", delimiter=';', names=['sentence', 'label'], header=None)

train_df2 = d2

test_df2 = pd.read_csv("archive2/test.txt", delimiter=';', names=['sentence', 'label'], header=None)


emotion2 = train_df2['label'].unique()
emotion_dict2 = {}
emotion_dict2 = {v: i for i, v in enumerate(emotion2)}

train_df2['label'] = train_df2['label'].apply(lambda x: emotion_dict2.get(x))
test_df2['label'] = test_df2['label'].apply(lambda x: emotion_dict2.get(x))

train_sentence2 = [sentence for sentence in train_df2['sentence']]
test_sentence2 = [sentence for sentence in test_df2['sentence']]

tokenizer = Tokenizer(oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentence2)
tokenizer.word_index

train_sequence2 = tokenizer.texts_to_sequences(train_sentence2)
train_padded2 = pad_sequences(train_sequence2, padding='post',
                             maxlen=max_length,
                             truncating=trunc_type)

test_sequence2 = tokenizer.texts_to_sequences(test_sentence2)
test_padded2 = pad_sequences(test_sequence2, padding='post', maxlen=max_length,
                            truncating=trunc_type)



Model_processing2 = model_processing(train_sentence=train_sentence2,
                                    test_sentence=test_sentence2)

Model_processing2.tokenizer_def()

train_label2 = []
for i in train_df2['label']:
    tmp = np.zeros(6)
    tmp[int(i)] = 1
    train_label2.append(tmp)
train_label2 = np.array(train_label)
test_label2 = []
for i in test_df2['label']:
    try:
        tmp = np.zeros(6)
        tmp[int(i)] = 1
        test_label2.append(tmp)
    except:
        continue

test_label = np.array(test_label)

train_padded = np.array(train_padded)
test_padded = np.array(test_padded)
Model_processing.main_model()
# Model_processing.fit_model(train_padded=train_padded,
#                            test_padded=test_padded,
#                            train_label=train_label,
#                            test_label=test_label)
swapped_emotion_dict = {value: key for key, value in emotion_dict.items()}
swapped_emotion_dict2 = {value: key for key, value in emotion_dict2.items()}
from answer import answer


def return_result(text):
    Model_processing.load_model("model_vn.keras")
    sentences = []
    sentences.append(text)
    sequence = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequence, maxlen=max_length,
                           padding=padding_type,
                           truncating=trunc_type)
    return answer(Model_processing.predict_model(padded=padded)[0], swapped_emotion_dict)


def return_result2(text):
    Model_processing2.load_model("model_new.h5")
    sentences = []
    sentences.append(text)
    sequence = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequence, maxlen=max_length,
                           padding=padding_type,
                           truncating=trunc_type)
    return answer(Model_processing2.predict_model(padded=padded)[0], swapped_emotion_dict2)