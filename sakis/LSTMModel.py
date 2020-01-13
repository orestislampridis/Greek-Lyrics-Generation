import os
import numpy as np
import config
from SongFeatures import SONG_BEGIN, SONG_PADDING, SONG_END
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, GRU, Bidirectional, Embedding
import tensorflow as tf


class LSTMModel:
    model = None
    callbacks = None
    transformer = None

    def __init__(self, vocabulary_size):
        dropout = 0.2
        self.model = Sequential()

        # Embedding layer
        self.model.add(Embedding(input_dim=vocabulary_size, output_dim=1024, mask_zero=True, trainable=True))

        # Bidirectional LSTM layer
        self.model.add(Bidirectional(LSTM(256, return_sequences=True)))

        # Bidirectional LSTM layer 2
        self.model.add(Bidirectional(LSTM(256, return_sequences=True)))

        # LSTM layer
        self.model.add(LSTM(256, return_sequences=True))

        # LSTM layer 2
        self.model.add(LSTM(256))

        if dropout > 0:
            self.model.add(Dropout(dropout))

        # Add dense layer
        self.model.add(Dense(vocabulary_size))

        # Add activation func
        self.model.add(Activation('softmax'))

        print("Compiling model...")
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

        # Name of the checkpoint files
        checkpoint_path = os.path.join(config.checkpoints_path, "checkpoint")
        checkpoint_path = os.path.normpath(checkpoint_path)

        checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor="val_accuracy", save_weights_only=True, verbose=1)
        print_callback = LambdaCallback(on_epoch_end=self.on_epoch_end)
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=20)
        self.callbacks = [checkpoint, print_callback, early_stopping]

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

    def on_epoch_end(self, epoch, logs):
        # Function invoked at end of each epoch. Prints generated text.
        print('\n----- Completed Epoch: %d\n' % (epoch+1))

        weights_path = os.path.join(config.checkpoints_path, "weights")
        self.model.save_weights(weights_path)

    # Data generator for fit and evaluate
    def generator(self, sequences):
        sequences_size = len(sequences)
        index = 0

        while True:
            x = np.zeros((config.batch_size, config.sequence_length), dtype=np.int32)
            y = np.zeros(config.batch_size, dtype=np.int32)

            for i in range(config.batch_size):
                row = sequences[index % sequences_size]

                for word_idx, word in enumerate(row[0]):
                    x[i, word_idx] = word
                y[i] = row[1]
                index = index + 1
            yield x, y

    def set_dataset_transformer(self, transformer):
        self.transformer = transformer

    def fit(self, train_data, test_data):
        train_steps = int(len(train_data) / config.batch_size) + 1
        validation_steps = int(len(test_data) / config.batch_size) + 1

        self.model.fit_generator(self.generator(train_data),
                                 steps_per_epoch=train_steps,
                                 epochs=config.epochs,
                                 callbacks=self.callbacks,
                                 validation_data=self.generator(test_data),
                                 validation_steps=validation_steps)

    def generate_text(self, transformer, start_word=SONG_BEGIN):
        # Text generation step (generating text using the learned model)
        max_length_words = 370

        # Converting our start string to numbers (vectorizing)
        input_eval = [transformer.encode_word(SONG_PADDING)] * config.sequence_length
        input_eval[-1] = transformer.encode_word(start_word)

        # Empty string to store our results
        text_generated = [start_word]

        # Low temperatures results in more predictable text.
        # Higher temperatures results in more surprising text.
        # Experiment to find the best setting.
        temperature = 1.0

        # Here batch size == 1
        self.model.reset_states()

        stop = False
        while not stop:
            if len(text_generated) >= max_length_words:
                text_generated.append(SONG_END)
                break

            predictions = self.model.predict(input_eval)[0]
            # apply temperature in log space
            predictions = np.asarray(predictions).astype('float64')
            predictions = np.log(predictions) / temperature
            predictions = np.exp(predictions)

            # normalize
            predictions = predictions / np.sum(predictions)
            probabilities = np.random.multinomial(1, predictions, 1)
            predicted_id = int(np.argmax(probabilities))

            # using a categorical distribution to predict the word returned by the model
            # predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

            # We pass the predicted word as the next input to the model
            # along with the previous hidden state
            next_sequence = []

            for idx in range(1, len(input_eval)):
                next_sequence.append(input_eval[idx])

            next_sequence.append(predicted_id)

            input_eval = next_sequence

            word = transformer.decode_id(predicted_id)
            text_generated.append(word)

            stop = word == SONG_END

        output_text = ""
        last_was_padding = False
        for word in text_generated:

            if word == SONG_PADDING:
                if not last_was_padding:
                    output_text += ". "
                    last_was_padding = True
                continue
            else:
                last_was_padding = False

            output_text += word
            output_text += " "

        return output_text
