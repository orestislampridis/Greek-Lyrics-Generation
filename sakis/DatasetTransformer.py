import SongFeatures as songs


class DatasetTransformer:
    words_to_ids = dict()
    ids_to_words = dict()
    vocabulary_frequencies = dict()

    def __init__(self, vocabulary):
        # create vocabulary frequencies (NOT USED)
        self.vocabulary_frequencies = songs.vocabulary_frequencies(vocabulary)
        # create word -> id and id -> word maps
        self.words_to_ids, self.ids_to_words = songs.map_vocabulary(vocabulary)

    def transform_encode(self, dataset):
        # encode dataset features and labels to ids
        return songs.transform_encode_dataset(dataset, self.words_to_ids)

    def transform_decode(self, dataset):
        # decode dataset features and labels to ids
        return songs.transform_decode_dataset(dataset, self.ids_to_words)

    def encode_word(self, word):
        try:
            return self.words_to_ids[word]
        except KeyError:  # treat out of vocabulary words as PADDING
            return self.words_to_ids[songs.SONG_PADDING]

    def decode_id(self, word_id):
        return self.ids_to_words[word_id]

    def encode_words(self, words):
        ids = []

        for word in words:
            word_id = self.encode_word(word)
            ids.append(word_id)

        return ids

    def decode_ids(self, ids):
        words = []

        for word_id in ids:
            word = self.decode_id(word_id)
            words.append(word)

        return words
