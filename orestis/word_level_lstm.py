from __future__ import print_function

import codecs
import io
import os
from create_lstm_model import create_lstm_model
import numpy as np

min_words = 2
sequence_length = 10
step = 1


def train_test_split(sentences_original, next_original, percentage_test=2):
    # shuffle at unison
    print('Shuffling sentences')

    temp_sentences = []
    temp_next_word = []
    for i in np.random.permutation(len(sentences_original)):
        temp_sentences.append(sentences_original[i])
        temp_next_word.append(next_original[i])

    cut_index = int(len(sentences_original) * (1.-(percentage_test/100.)))
    x_train, x_test = temp_sentences[:cut_index], temp_sentences[cut_index:]
    y_train, y_test = temp_next_word[:cut_index], temp_next_word[cut_index:]

    print("Size of training set = %d" % len(x_train))
    print("Size of test set = %d" % len(y_test))
    return (x_train, y_train), (x_test, y_test)


def write_vocabulary_file(words_file_path, words_set):
    words_file = codecs.open(words_file_path, 'w', encoding='utf8')
    for w in words_set:
        if w != "\n":
            words_file.write(w+"\n")
        else:
            words_file.write(w)
    words_file.close()


def get_text_data(data_path, vocab_file_name):
    sentences = []
    next_words = []
    ignored = 0

    with io.open(data_path, encoding='utf-8') as f:
        input_text = f.read().lower().replace('\n', ' \n ')
    print('Corpus length in characters:', len(input_text))

    text_in_words = [w for w in input_text.split(' ') if w.strip() != '' or w == '\n']
    print('Corpus length in words:', len(text_in_words))

    # Calculate word frequency
    word_freq = {}
    for word in text_in_words:
        word_freq[word] = word_freq.get(word, 0) + 1

    ignored_words = set()
    for k, v in word_freq.items():
        if word_freq[k] < min_words:
            ignored_words.add(k)

    words = set(text_in_words)
    print('Number of unique words before ignoring:', len(words))
    print('Ignoring words with frequency <', min_words)
    words = sorted(set(words) - ignored_words)
    print('Number of unique words after ignoring:', len(words))

    write_vocabulary_file(vocab_file_name, words)

    word_index = dict((c, i) for i, c in enumerate(words))
    reverse_word_index = dict((i, c) for i, c in enumerate(words))

    # cut the text in semi-redundant sequences of SEQUENCE_LEN words
    for i in range(0, len(text_in_words) - sequence_length, step):
        # Only add sequences where no word is in ignored_words
        if len(set(text_in_words[i: i + sequence_length + 1]).intersection(ignored_words)) == 0:
            sentences.append(text_in_words[i: i + sequence_length])
            next_words.append(text_in_words[i + sequence_length])
        else:
            ignored = ignored + 1

    print('Ignored sequences:', ignored)
    print('Number of sequences after ignoring:', len(sentences))

    return sentences, words, next_words, word_index, reverse_word_index


if __name__ == "__main__":

    data_path = "data/entexna.txt"
    vocab_file = "data/vocab.txt"

    text = get_text_data(data_path=data_path, vocab_file_name=vocab_file)

    sentences, words, next_words, word_index, reverse_word_index = text

    (sentences, next_words), (sentences_test, next_words_test) = train_test_split(sentences, next_words)

    if not os.path.isdir('./checkpoints/'):
        os.makedirs('./checkpoints/')

    if not os.path.isdir('./data/'):
        os.makedirs('./data/')

    model = create_lstm_model(words=words, sentences=sentences, next_words=next_words, sentences_test=sentences_test,
                              next_words_test=next_words_test, word_index=word_index, reverse_word_index=reverse_word_index,
                              min_words=min_words, sequence_length=sequence_length, step=step)
