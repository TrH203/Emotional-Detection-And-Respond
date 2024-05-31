
import torch
import transformer_utils
import tensorflow as tf
import numpy as np
import time
import sentencepiece as spm
tokenizer = spm.SentencePieceProcessor(model_file='sentencepiece.model')

num_layers = 2
embedding_dim = 128
fully_connected_dim = 128
num_heads = 2
positional_encoding_length = 256

encoder_vocab_size = int(tokenizer.vocab_size())
decoder_vocab_size = encoder_vocab_size

# Initialize the model
transformer = transformer_utils.Transformer(
    num_layers, 
    embedding_dim, 
    num_heads,
    fully_connected_dim,
    encoder_vocab_size, 
    decoder_vocab_size, 
    positional_encoding_length, 
    positional_encoding_length,
)

transformer.load_weights('./weights/F1')


def answer_question(question, model, tokenizer, encoder_maxlen=250, decoder_maxlen=150):
    # Tokenize the question
    tokenized_question = tokenizer.encode_as_ids("Question: " + question)

    # Add an extra dimension to the tensor
    tokenized_question = tf.expand_dims(tokenized_question, 0)

    # Pad the question tensor
    padded_question = tf.keras.preprocessing.sequence.pad_sequences(tokenized_question,
                                                                    maxlen=encoder_maxlen,
                                                                  padding='post',
                                                                    truncating='post')
    tokenized_answer = tokenizer.encode_as_ids("answer: ")

    tokenized_answer = tf.expand_dims(tokenized_answer, 0)
    eos = tokenizer.piece_to_id(['</s>'])

    for i in range(decoder_maxlen):

        next_word = transformer_utils.next_word(padded_question, tokenized_answer, model)

        tokenized_answer = tf.concat([tokenized_answer, next_word], axis=1)

        if next_word == eos:
            break

    return tokenized_answer

# result = answer_question("Question: Now I have anger and trust issues. How can I treat this and fix myself?", transformer, tokenizer)
# print(result[0].numpy().tolist())
# print("----DU DOAN----")
# print(tokenizer.decode_ids(result[0].numpy().tolist()))