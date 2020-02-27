from __future__ import print_function

import os
import gc
import sys
import config
import Persistence as io
from DatasetTransformer import DatasetTransformer
from LSTMModel import LSTMModel
from LyricsDataset import LyricsDataset
from SongFeatures import split_train_test, SONG_BEGIN


def train_main():
    print("Initializing...")

    # prepare environment
    if not os.path.isdir('./checkpoints/'):
        os.makedirs('./checkpoints/')

    if not os.path.isdir('./data/'):
        os.makedirs('./data/')

    print("Creating model...")
    # load dataset
    lyrics_dataset = LyricsDataset()
    # create word <-> id encoder/decoder
    data_transformer = DatasetTransformer(lyrics_dataset.vocabulary_info)

    # create model instance
    lyrics_model = LSTMModel(lyrics_dataset.vocabulary_size())
    lyrics_model.set_dataset_transformer(data_transformer)

    print("Preparing dataset...")
    # encode words to ids
    dataset = data_transformer.transform_encode(lyrics_dataset.dataset)
    # split dataset to train and test sequences (not used)
    #train_data, test_data = split_train_test(dataset, test_data_ratio=0.3)

    print("Training...")
    # train the model on ALL the sequences and evaluate
    # on the same data to force the RNN to learn all the
    # sequences, this is a generation effort after all
    lyrics_model.fit(dataset, dataset)

    # train the model on the train sequences
    # after the first few epochs train on train sequences
    #lyrics_model.fit(train_data, test_data)

    return 0


def generate_main(start_seed=None):
    # if no seed was found create a default empty sequence
    if start_seed is None:
        start_seed = [SONG_BEGIN]

    print("Initializing...")

    # prepare environment
    if not os.path.isdir('./checkpoints/'):
        print("LSTM model must be trained first.")
        exit(1)

    if not os.path.isdir('./data/'):
        print("LSTM model must be trained first.")
        exit(1)

    print("Restoring vocabulary and model...")
    vocabulary_info = io.read_vocabulary_info_json(config.vocabulary_path)
    data_transformer = DatasetTransformer(vocabulary_info)

    # create model instance (vocabulary length + 3 is for synthetic words like SONG_BEGIN)
    lyrics_model = LSTMModel(len(vocabulary_info) + 3)
    lyrics_model.set_dataset_transformer(data_transformer)

    print("Generating...")

    text = lyrics_model.generate_text(data_transformer, start_seed)
    print("Generated Song:\n", text, "\n")

    return 0


def args_error():
    print("Illegal argument error. Use --train or --gen [seed]")
    exit(1)


if __name__ == "__main__":
    arg_count = len(sys.argv)

    if arg_count < 2:
        args_error()

    run_option_switch = sys.argv[1]
    do_train = run_option_switch == "--train"
    do_generate = run_option_switch == "--gen"

    if not do_train and not do_generate:
        args_error()

    if do_train and arg_count != 2:
        args_error()

    if do_train:
        print("Greek Lyrics - Generator Model Training.")

        # due to memory concerns train to the target epoch
        # iteratively to avoid out of memory errors
        target_epochs = 100
        counter = 0
        stop = False

        while not stop:
            # force garbage collection
            gc.collect()
            # run train loop
            train_main()

            counter += 1
            stop = counter * config.epochs >= target_epochs
    else:
        print("Greek Lyrics - Generator.")
        seed = [SONG_BEGIN]

        # add args to seed sequence
        for arg_idx in range(2, len(sys.argv)):
            seed.append(sys.argv[arg_idx])

        # run prediction loop
        generate_main(seed)
