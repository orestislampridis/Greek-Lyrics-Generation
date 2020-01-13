import SongFeatures as songs


class DatasetTransformer:
    words_to_ids = dict()
    ids_to_words = dict()

    def __init__(self, vocabulary):
        self.words_to_ids, self.ids_to_words = songs.map_vocabulary(vocabulary)

    def transform_encode(self, dataset):
        return songs.transform_encode_dataset(dataset, self.words_to_ids)

    def transform_decode(self, dataset):
        return songs.transform_decode_dataset(dataset, self.ids_to_words)

    def encode_word(self, word):
        return self.words_to_ids[word]

    def decode_id(self, word_id):
        return self.ids_to_words[word_id]

    def encode_words(self, words):
        ids = []

        for word in words:
            word_id = self.words_to_ids[word]
            ids.append(word_id)

        return ids

    def decode_ids(self, ids):
        words = []

        for word_id in ids:
            word = self.ids_to_words[word_id]
            words.append(word)

        return words
