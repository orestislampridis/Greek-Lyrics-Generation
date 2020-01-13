import SongFeatures as songs
import config
import os
import Persistence as io


class LyricsDataset:
    vocabulary_info = None
    lyrics = None
    dataset = None

    def __init__(self):
        self.lyrics, self.vocabulary_info = songs.read_lyrics("data/lyrics.csv")

        if not os.path.exists(config.vocabulary_path):
            io.save_vocabulary_info_json(self.vocabulary_info, config.vocabulary_path)

        self.dataset = songs.lyrics_to_dataset(self.lyrics, config.sequence_length)

    def vocabulary_size(self):
        return len(self.vocabulary_info) + 3
