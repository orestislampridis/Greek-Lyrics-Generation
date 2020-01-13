import os
import string as s
import pandas as pd
import numpy as np

try:
    import nltk
except ImportError:
    print("ERROR: NLTK is required.")
    print("Please install NLTK using the command:")
    print("$ pip install -U nltk")
    exit(1)
try:
    import spacy
except ImportError:
    print("ERROR: SpaCy is required.")
    print("Please install SpaCy using the command:")
    print("$ pip install -U spacy")
    exit(1)

SONG_BEGIN = "[BEGIN]"
SONG_PADDING = "[PAD]"
SONG_END = "[END]"


def read_lyrics_from_string(string):
    try:
        nlp = spacy.load("el_core_news_sm")
    except Exception:
        print("ERROR: SpaCy greek language dependencies are required.")
        print("Install SpaCy greek dependencies by using the following commands: ")
        print("$ python -m spacy download el_core_news_sm")
        print("$ python -m spacy download el_core_news_md")
        exit(1)

    sanitized = string.replace("\n", " ")

    doc = nlp(sanitized)
    lyrics = []

    for sentence in doc.sents:
        text = sentence.text
        words = nltk.word_tokenize(text, language="greek")

        # remove punctuation
        words = list(filter(lambda it: (it.isdigit() or it.isalpha()), words))
        # to lowercase
        words = list(map(lambda it: it.lower(), words))

        lyrics.append(words)

    return lyrics


def read_lyrics(path, delimiter='\t', encoding='UTF-8'):
    # Install help for spacy
    # pip install -U spacy
    # conda install -c conda-forge spacy
    # python -m spacy download el_core_news_sm
    # python -m spacy download el_core_news_md

    try:
        nlp = spacy.load("el_core_news_sm")
    except Exception:
        print("ERROR: SpaCy greek language dependencies are required.")
        print("Install SpaCy greek dependencies by using the following commands: ")
        print("$ python -m spacy download el_core_news_sm")
        print("$ python -m spacy download el_core_news_md")
        exit(1)

    data = pd.read_csv(path, delimiter=delimiter, encoding=encoding)

    sample_size = len(data.values)
    vocabulary_info = dict()
    songs = []

    # read song lyrics + tokenize
    for (doc_idx, row) in enumerate(data.values):
        song_lyrics = row[5]

        if song_lyrics[-1] != ".":
            song_lyrics = song_lyrics + "."

        doc = nlp(song_lyrics)

        sents = []

        for sentence in doc.sents:
            text = sentence.text
            words = nltk.word_tokenize(text, language="greek")

            # remove punctuation
            words = list(filter(lambda it: (it.isdigit() or it.isalpha()), words))
            # to lowercase
            words = list(map(lambda it: it.lower(), words))

            # count word frequencies
            for word in words:

                if word not in vocabulary_info.keys():
                    vocabulary_info[word] = [0] * sample_size

                word_frequencies = vocabulary_info[word]
                word_frequencies[doc_idx] += 1
                vocabulary_info[word] = word_frequencies

            sents.append(words)

        songs.append(sents)

    return songs, vocabulary_info


def lyrics_to_dataset(lyrics, sequence_length):
    flattened_songs = []  # [[], ""]

    for song in lyrics:
        song_words = [SONG_BEGIN]

        for sent in song:
            for word in sent:
                song_words.append(word)

        song_words.append(SONG_END)
        flattened_songs.append(song_words)

    feature_dataset = []

    for song in flattened_songs:
        # init sequence and next word (X - Y features)
        sequence = []

        for word in song:

            if len(sequence) < sequence_length:
                sequence.append(word)
            else:
                feature_dataset.append((sequence, word))
                sequence = [word]

        if len(sequence) == 0:
            continue

        if len(sequence) < sequence_length and sequence[0] != SONG_END:
            while len(sequence) < sequence_length:
                sequence.append(SONG_PADDING)

            feature_dataset.append((sequence, SONG_PADDING))

    return feature_dataset


def map_vocabulary(vocabulary):
    words_map = dict()
    ids_map = dict()

    # map synthetic words
    words_map[SONG_PADDING] = 0
    ids_map[0] = SONG_PADDING

    words_map[SONG_BEGIN] = 1
    ids_map[1] = SONG_BEGIN

    words_map[SONG_END] = 2
    ids_map[2] = SONG_END

    for idx, word in enumerate(sorted(vocabulary.keys())):
        word_id = 3 + idx
        words_map[word] = word_id
        ids_map[word_id] = word

    return words_map, ids_map


def transform_encode_dataset(dataset, words_map):
    transformed = []

    for row in dataset:
        sequence = row[0]
        next_word = row[1]

        transformed_sequence = []
        transformed_next_word = words_map[next_word]

        for word in sequence:
            transformed_sequence.append(words_map[word])

        transformed.append((transformed_sequence, transformed_next_word))

    return transformed


def transform_decode_dataset(dataset, ids_map):
    transformed = []

    for row in dataset:
        sequence = row[0]
        next_word_id = row[1]

        transformed_sequence = []
        transformed_next_word = ids_map[next_word_id]

        for word_id in sequence:
            transformed_sequence.append(ids_map[word_id])

        transformed.append((transformed_sequence, transformed_next_word))

    return transformed


def split_train_test(dataset, test_data_ratio=0.3, seed=None):
    if seed is not None:
        np.random.seed(seed)

    data_size = len(dataset)
    test_data_size = np.floor(data_size * test_data_ratio)
    shuffled_indices = np.random.permutation(data_size)

    train_data = []
    test_data = []

    for idx in shuffled_indices:
        row = dataset[idx]

        if len(test_data) < test_data_size:
            test_data.append(row)
        else:
            train_data.append(row)

    return train_data, test_data
