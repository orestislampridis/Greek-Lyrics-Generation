import numpy as np
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional
from keras.models import Sequential

dropout = 0.2
batch_size = 32
output = 'output.txt'


def create_lstm_model(words, sentences, next_words, sentences_test, next_words_test, word_index, reverse_word_index,
                      min_words, sequence_length, step):

    def generator(sentence_list, next_word_list, batch_size):
        index = 0
        while True:
            x = np.zeros((batch_size, sequence_length, len(words)), dtype=np.bool)
            y = np.zeros((batch_size, len(words)), dtype=np.bool)
            for i in range(batch_size):
                for t, w in enumerate(sentence_list[index]):
                    x[i, t, word_index[w]] = 1
                y[i, word_index[next_word_list[index]]] = 1

                index = index + 1
                if index == len(sentence_list):
                    index = 0
            yield x, y

    def sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def on_epoch_end(epoch, logs):
        # Function invoked at end of each epoch. Prints generated text.
        output_file.write('\n----- Generating text after Epoch: %d\n' % epoch)

        # Randomly pick a seed sequence
        seed_index = np.random.randint(len(sentences + sentences_test))
        seed = (sentences + sentences_test)[seed_index]

        for diversity in [0.3, 0.4, 0.5, 0.6, 0.7]:
            sentence = seed
            output_file.write('----- Diversity:' + str(diversity) + '\n')
            output_file.write('----- Generating with seed:\n"' + ' '.join(sentence) + '"\n')
            output_file.write(' '.join(sentence))

            for i in range(50):
                x_pred = np.zeros((1, sequence_length, len(words)))
                for t, word in enumerate(sentence):
                    x_pred[0, t, word_index[word]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_word = reverse_word_index[next_index]

                sentence = sentence[1:]
                sentence.append(next_word)

                output_file.write(" " + next_word)
            output_file.write('\n')
        output_file.write('=' * 80 + '\n')
        output_file.flush()

    model = Sequential()
    model.add(Bidirectional(LSTM(256), input_shape=(sequence_length, len(words))))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(len(words)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    print(model.summary())

    file_path = "./checkpoints/LSTM_LYRICS-epoch{epoch:03d}-words%d" \
                "-sequence%d-minfreq%d-loss{loss:.4f}-acc{acc:.4f}-" \
                "val_loss{val_loss:.4f}-val_acc{val_acc:.4f}" % (
                    len(words),
                    sequence_length,
                    min_words)

    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', save_best_only=True)
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    early_stopping = EarlyStopping(monitor='val_acc', patience=5)
    callbacks_list = [checkpoint, print_callback, early_stopping]

    output_file = open(output, "w", encoding="utf-8")
    model.fit_generator(generator(sentences, next_words, batch_size),
                        steps_per_epoch=int(len(sentences) / batch_size) + 1,
                        epochs=100,
                        callbacks=callbacks_list,
                        validation_data=generator(sentences_test, next_words_test, batch_size),
                        validation_steps=int(len(sentences_test) / batch_size) + 1)

    return model
