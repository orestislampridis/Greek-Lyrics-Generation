"""
Based on
https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
https://github.com/enriqueav/lstm_lyrics/blob/master/lstm_train_embedding.py
"""

import os
import numpy as np
import random
import config
from SongFeatures import SONG_BEGIN, SONG_PADDING, SONG_END
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, GRU, Bidirectional, Embedding, Flatten
import tensorflow as tf


class LSTMModel:
    vocabulary_size = 0
    model = None
    callbacks = None
    transformer = None

    def __init__(self, vocabulary_size):
        dropout = 0.2
        self.vocabulary_size = vocabulary_size

        # init keras model
        self.model = Sequential()

        # add embedding layer, large vectors allow for more fine grained
        # relationships to be inferred during training
        self.model.add(Embedding(input_dim=vocabulary_size,
                                 output_dim=1024,
                                 mask_zero=True,
                                 trainable=True, batch_input_shape=[config.batch_size, None]))

        # add Bi-LSTM layer, with state
        self.model.add(Bidirectional(LSTM(128, return_sequences=False,
                                          return_state=False,
                                          stateful=True,
                                          recurrent_initializer="glorot_uniform")))

        # add dropout
        self.model.add(Dropout(dropout))

        # add dense layer, output size is the same as number of possible words
        self.model.add(Dense(vocabulary_size))

        # add activation, softmax for probabilities
        self.model.add(Activation("softmax"))

        print("Compiling model...")
        # - sparse_categorical_crossentropy is used for NON one-hot encodings
        # - adam and adamax are the best optimizers for this case,
        # adamax is better for embeddings during first few epochs
        # - sparse_categorical_accuracy is an accuracy metric for
        # categorical dist outputs, that are NOT one-hot encoded
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",
                           metrics=["sparse_categorical_accuracy"])

        # tf.keras.utils.plot_model(
        #     self.model,
        #     to_file='model.png',
        #     show_shapes=False,
        #     show_layer_names=False,
        #     rankdir='TB',
        #     expand_nested=True,
        #     dpi=300
        # )

        # name of the checkpoint files
        checkpoint_path = os.path.join(config.checkpoints_path, "checkpoint")
        checkpoint_path = os.path.normpath(checkpoint_path)

        checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                     monitor="sparse_categorical_accuracy",
                                     save_weights_only=True,
                                     verbose=1)
        print_callback = LambdaCallback(on_epoch_end=self.on_epoch_end)
        early_stopping = EarlyStopping(monitor='sparse_categorical_accuracy', patience=3)
        self.callbacks = [checkpoint, print_callback, early_stopping]

        # used as a helper in order to restore to an earlier checkpoint
        weights_path = os.path.join(config.checkpoints_path, "weights")

        if os.path.exists(weights_path):
            print("Restoring from weights...")
            self.model.load_weights(weights_path)
        else:
            print("Restoring from checkpoint...")
            checkpoints_dir = os.path.normpath(config.checkpoints_path)
            latest_checkpoint = tf.train.latest_checkpoint(checkpoints_dir)

            if latest_checkpoint is not None:
                self.model.load_weights(latest_checkpoint)

    # epoch end callback
    def on_epoch_end(self, epoch, logs):
        # Function invoked at end of each epoch. Prints generated text.
        print('\n----- Completed Epoch: %d\n' % (epoch + 1))

    # generator retrieves sequences for fit and evaluate
    def generator(self, sequences, is_test=False):
        sequences_size = len(sequences)
        # use two indices, to separate train/test sequence retrieval
        train_index = 0
        test_index = 0

        x = np.zeros((config.batch_size, config.sequence_length), dtype=np.int32)
        y = np.zeros(config.batch_size, dtype=np.int32)

        while True:
            for i in range(config.batch_size):

                if is_test:
                    index = test_index
                    test_index += 1
                else:
                    index = train_index
                    train_index += 1

                row = sequences[index % sequences_size]

                # add row data to x, y
                for word_idx, word in enumerate(row[0]):
                    x[i, word_idx] = word

                y[i] = row[1]

                # reset index if needed, prevent num overflow
                # because index++ will go on forever while training
                # should not be problem but still rest as sanity check
                if index % sequences_size == sequences_size-1:
                    if is_test:
                        test_index = 0
                    else:
                        train_index = 0

            # yield until next call
            yield x, y

    # pass the dataset transformer to encode/decode input and output
    def set_dataset_transformer(self, transformer):
        self.transformer = transformer

    # fit the model
    def fit(self, train_data, test_data):
        train_steps = int(len(train_data) / config.batch_size) + 1
        validation_steps = int(len(test_data) / config.batch_size) + 1

        # fit using generator method
        self.model.fit_generator(self.generator(train_data, is_test=False),
                                 steps_per_epoch=train_steps,
                                 epochs=config.epochs,
                                 callbacks=self.callbacks,
                                 validation_data=self.generator(test_data, is_test=True),
                                 validation_steps=validation_steps)

    # generate text using a trained model
    def generate_text(self, transformer, seed=SONG_BEGIN):
        # generate a random upper bound for generated text
        max_length_words = int(random.uniform(3 * config.sequence_length, 185))

        # Init seed sequence
        seed_sequence = [transformer.encode_word(SONG_PADDING)] * config.sequence_length

        # Empty collection to store results
        text_generated = []

        # fill seed sequence
        if isinstance(seed, str):
            seed_sequence[-1] = transformer.encode_word(seed)
            text_generated = [seed]
        elif isinstance(seed, list):
            added_count = 0
            input_sequence_idx = -1

            # -1 for reversal, might be a better way
            for idx in range(len(seed) - 1, -1, -1):

                if added_count >= len(seed_sequence):
                    break

                word = seed[idx]

                if word in [SONG_BEGIN, SONG_END, SONG_PADDING]:
                    seed_sequence[input_sequence_idx] = transformer.encode_word(word)
                else:
                    seed_sequence[input_sequence_idx] = transformer.encode_word(word.lower())

                added_count += 1
                input_sequence_idx -= 1

            excluded_words = [SONG_BEGIN, SONG_END, SONG_PADDING]

            for word in seed:
                if word in excluded_words:
                    continue

                text_generated.append(word)

        vocabulary_frequencies = transformer.vocabulary_frequencies

        SAMPLE_NONE = 'sample_none'  # uses max probability with NO sampling + token frequency
        SAMPLE_CATEGORICAL = 'sample_categorical'  # uses sampled probability with categorical dist
        SAMPLE_MULTINOMIAL = 'sample_multinomial'  # uses sampled probability with multinomial dist

        # sampling method
        sample_method = SAMPLE_MULTINOMIAL

        # Use high temperature to show a more readable first output.
        temperature = 1

        # reset states if needed
        self.model.reset_states()

        stop = False

        while not stop:
            # force stop when upper text bound is reached
            if len(text_generated) >= max_length_words:
                text_generated.append(SONG_END)
                break

            seed_batch = [seed_sequence] * config.batch_size
            predictions = self.model.predict(seed_batch, batch_size=config.batch_size)

            if sample_method == SAMPLE_NONE:
                # use argmax + word frequency to predict the next word
                predictions = predictions[0] / temperature
                predicted_id = int(self.__probable_word(transformer, vocabulary_frequencies, predictions))
            elif sample_method == SAMPLE_CATEGORICAL:
                # using a categorical distribution to predict the word returned by the model
                predictions = predictions / temperature
                predicted_id = int(tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy())
            elif sample_method == SAMPLE_MULTINOMIAL:
                predictions = np.asarray(predictions[0]).astype('float64')

                # apply temperature in log space
                predictions = np.log(predictions) / temperature
                exp_predictions = np.exp(predictions)
                predictions = exp_predictions / np.sum(exp_predictions)
                probabilities = np.random.multinomial(1, predictions, 1)
                predicted_id = int(np.argmax(probabilities))
            else:
                raise ValueError("Invalid sampling method selected.")

            # remap the input sequence - context still remains in LSTM states
            next_sequence = []

            for idx in range(1, len(seed_sequence)):
                next_sequence.append(seed_sequence[idx])

            next_sequence.append(predicted_id)

            seed_sequence = next_sequence

            word = transformer.decode_id(predicted_id)
            text_generated.append(word)

            # adjust temperature to randomize output
            # use low to medium temperature for readable results
            temperature = 0.25 + random.uniform(0, 0.25)

            stop = word == SONG_END

        # convert generated sequence to string
        output_text = ""
        last_was_padding = False
        for word in text_generated:
            # do not output begin and end tokens
            if word == SONG_BEGIN or word == SONG_END:
                continue

            # if it is a padding token then add period and skip next padding tokens
            if word == SONG_PADDING:
                if not last_was_padding:
                    output_text += ". "
                    last_was_padding = True
                continue
            else:
                last_was_padding = False

            output_text += word
            output_text += " "

        # post processing (just minor punctuation cleanup)
        cleaned = output_text.replace(" . ", ". ")
        return cleaned

    # ad hoc way to find next probable word
    def __probable_word(self, transformer, vocabulary_freq, predictions):
        padding_id = transformer.encode_word(SONG_PADDING)

        max_proba_idx = predictions.argmax()
        max_proba = predictions[max_proba_idx]

        candidate_ids = []

        # find probable words with a margin of 0.001
        for word_id, proba in enumerate(predictions):
            if proba >= max_proba-0.001:
                candidate_ids.append(word_id)

        if len(candidate_ids) == 0:
            return padding_id
        elif len(candidate_ids) == 1:
            return candidate_ids[0]

        max_freq = -1
        selected_word_id = padding_id

        # return the most frequent word
        for word_id in candidate_ids:
            word = transformer.decode_id(word_id)
            freq = 0

            if word not in [SONG_BEGIN, SONG_END, SONG_PADDING]:
                freq = vocabulary_freq[word]
            else:
                freq = 1150

            if freq >= max_freq:
                selected_word_id = word_id

        return selected_word_id
