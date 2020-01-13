from __future__ import print_function

import os
import config
import Persistence as io
from DatasetTransformer import DatasetTransformer
from LSTMModel import LSTMModel
from LyricsDataset import LyricsDataset
from SongFeatures import split_train_test
import tensorflow.compat.v1 as tf


def train_main():
    print("Initializing...")

    # prepare environment
    # Assume that the number of cores per socket in the machine is denoted as NUM_PARALLEL_EXEC_UNITS
    #  when NUM_PARALLEL_EXEC_UNITS=0 the system chooses appropriate settings
    # NUM_PARALLEL_EXEC_UNITS = 8
    # tf_config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS,
    #                            inter_op_parallelism_threads=8,
    #                            allow_soft_placement=True,
    #                            device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})
    #
    # session = tf.Session(config=tf_config)

    if not os.path.isdir('./checkpoints/'):
        os.makedirs('./checkpoints/')

    if not os.path.isdir('./data/'):
        os.makedirs('./data/')

    print("Creating model...")
    lyrics_dataset = LyricsDataset()
    data_transformer = DatasetTransformer(lyrics_dataset.vocabulary_info)
    lyrics_model = LSTMModel(lyrics_dataset.vocabulary_size())
    lyrics_model.set_dataset_transformer(data_transformer)

    print("Preparing dataset...")
    dataset = data_transformer.transform_encode(lyrics_dataset.dataset)
    train_data, test_data = split_train_test(dataset, test_data_ratio=0.2, seed=8)

    print("Training...")
    lyrics_model.fit(train_data, test_data)

    text = lyrics_model.generate_text(data_transformer)
    print("\n\n\nDONE\n\n\nGenerated text:\n", text)
    return 0


def generate_main():
    print("Initializing...")

    # prepare environment
    if not os.path.isdir('./checkpoints/'):
        print("LSTM model must be trained first.")
        exit(1)

    if not os.path.isdir('./data/'):
        print("LSTM model must be trained first.")
        exit(1)

    print("Restoring model...")
    vocabulary_info = io.read_vocabulary_info_json(config.vocabulary_path)
    data_transformer = DatasetTransformer(vocabulary_info)
    lyrics_model = LSTMModel(len(vocabulary_info) + 3)
    lyrics_model.set_dataset_transformer(data_transformer)

    print("Generating text...")
    text = lyrics_model.generate_text(data_transformer)
    print("Song:", text)

    return 0


if __name__ == "__main__":
    train_main()
    # generate_main()
