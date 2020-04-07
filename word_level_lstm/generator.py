"""
Script to generate text from an already trained network
It is necessary to at least provide the trained model and the vocabulary file
Based on
https://github.com/enriqueav/lstm_lyrics/blob/master/generate.py
"""

import argparse
import numpy as np
import re
from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional
from keras.models import Sequential

dropout = 0.2
batch_size = 32


def validate_seed(vocabulary, seed):
    # Validate that all the words in the seed are part of the vocabulary
    print("\nValidating that all the words in the seed are part of the vocabulary: ")
    seed_words = seed.split(" ")
    valid = True
    for w in seed_words:
        if w not in vocabulary:
            valid = False
    return valid


# Function from keras-team/keras/blob/master/examples/lstm_text_generation.py
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(model, reverse_word_index, word_index, seed, sequence_length, diversity, quantity):

    sentence = seed.split(" ")

    output = "generated.txt"
    output_file = open(output, "a", encoding="utf-8")

    output_file.write('Diversity:' + str(diversity) + '\n')
    output_file.write('Generating with seed:\n"' + ' '.join(sentence) + '"\n')
    output_file.write(' '.join(sentence))

    for i in range(quantity):
        x_pred = np.zeros((1, sequence_length, len(vocabulary)))
        for t, word in enumerate(sentence):
            x_pred[0, t, word_index[word]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_word = reverse_word_index[next_index]

        sentence = sentence[1:]
        sentence.append(next_word)

        output_file.write(" " + next_word)
    output_file.write('\n')


# Should be exactly the same model as in create_lstm_model.py
def create_model(len_words):
    keras_model = Sequential()
    keras_model.add(Bidirectional(LSTM(256), input_shape=(sequence_length, len_words)))
    if dropout > 0:
        keras_model.add(Dropout(dropout))
    keras_model.add(Dense(len_words))
    keras_model.add(Activation('softmax'))

    keras_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    return keras_model


if __name__ == "__main__":

    # The path of vocabulary used by the network
    vocabulary_file = "./data/vocab.txt"
    # The trained LSTM model
    model_checkpoint = "./checkpoints/LSTM_LYRICS-epoch032-words8002-sequence10-minfreq2-loss0.0768" \
                 "-acc0.9819-val_loss6.8975-val_acc0.3589"
    # The seed that will be used to generate the text
    seed = "Πως έγινε και αλλάξαμε πορεία και μένει το όνειρο ξανά στην ψυχή μου"
    seed = seed.lower()
    # The sequence length that was used for training. Only this number of words from the seed will be used
    sequence_length = 10
    # The value of diversity
    diversity = 1
    # Number of words to generate
    quantity = 50

    vocabulary = open(vocabulary_file, "r", encoding='utf8').readlines()
    # remove the \n at the end of the word, except for the \n word itself
    vocabulary = [re.sub(r'(\S+)\s+', r'\1', w) for w in vocabulary]
    vocabulary = sorted(set(vocabulary))

    word_index = dict((c, i) for i, c in enumerate(vocabulary))
    reverse_word_index = dict((i, c) for i, c in enumerate(vocabulary))

    # Create a basic model instance
    model = create_model(len(vocabulary))

    model.load_weights(model_checkpoint)
    model.summary()

    if validate_seed(vocabulary, seed):
        print("Seed is correct. All words you selected are part of the vocabulary.")
        # repeat the seed in case is not long enough, and take only the last elements
        seed = " ".join((((seed+" ")*sequence_length)+seed).split(" ")[-sequence_length:])
        generate_text(
            model, reverse_word_index, word_index, seed, sequence_length, diversity, quantity
        )
    else:
        print("Not all the words you selected are part of the vocabulary. Please provide different ones.")
        exit(0)
