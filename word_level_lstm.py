from __future__ import print_function

import io

import numpy as np
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional
from keras.models import Sequential

dropout = 0.2
min_words = 5
sequence_length = 10
step = 1
batch_size = 32


def shuffle_and_split_training_set(sentences_original, next_original, percentage_test=10):
    # shuffle at unison
    print('Shuffling sentences')

    tmp_sentences = []
    tmp_next_word = []
    for i in np.random.permutation(len(sentences_original)):
        tmp_sentences.append(sentences_original[i])
        tmp_next_word.append(next_original[i])

    cut_index = int(len(sentences_original) * (1.-(percentage_test/100.)))
    x_train, x_test = tmp_sentences[:cut_index], tmp_sentences[cut_index:]
    y_train, y_test = tmp_next_word[:cut_index], tmp_next_word[cut_index:]

    print("Size of training set = %d" % len(x_train))
    print("Size of test set = %d" % len(y_test))
    return (x_train, y_train), (x_test, y_test)


def generator(sentence_list, next_word_list, batch_size):
    index = 0
    while True:
        x = np.zeros((batch_size, sequence_length, len(words)), dtype=np.bool)
        y = np.zeros((batch_size, len(words)), dtype=np.bool)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index]):
                x[i, t, word_indices[w]] = 1
            y[i, word_indices[next_word_list[index]]] = 1

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
    seed_index = np.random.randint(len(sentences+sentences_test))
    seed = (sentences+sentences_test)[seed_index]

    for diversity in [0.3, 0.4, 0.5, 0.6, 0.7]:
        sentence = seed
        output_file.write('----- Diversity:' + str(diversity) + '\n')
        output_file.write('----- Generating with seed:\n"' + ' '.join(sentence) + '"\n')
        output_file.write(' '.join(sentence))

        for i in range(50):
            x_pred = np.zeros((1, sequence_length, len(words)))
            for t, word in enumerate(sentence):
                x_pred[0, t, word_indices[word]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_word[next_index]

            sentence = sentence[1:]
            sentence.append(next_word)

            output_file.write(" "+next_word)
        output_file.write('\n')
    output_file.write('='*80 + '\n')
    output_file.flush()


output = 'output.txt'
filename = 'entexna.txt'
with io.open(filename, encoding='utf-8') as f:
    text = f.read().lower().replace('\n', ' \n ')
print('Corpus length in characters:', len(text))

text_in_words = [w for w in text.split(' ') if w.strip() != '' or w == '\n']
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
print('Unique words before ignoring:', len(words))
print('Ignoring words with frequency <', min_words)
words = sorted(set(words) - ignored_words)
print('Unique words after ignoring:', len(words))

word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))

# cut the text in semi-redundant sequences of SEQUENCE_LEN words
sentences = []
next_words = []
ignored = 0
for i in range(0, len(text_in_words) - sequence_length, step):
    # Only add sequences where no word is in ignored_words
    if len(set(text_in_words[i: i+sequence_length+1]).intersection(ignored_words)) == 0:
        sentences.append(text_in_words[i: i + sequence_length])
        next_words.append(text_in_words[i + sequence_length])
    else:
        ignored = ignored+1
print('Ignored sequences:', ignored)
print('Remaining sequences:', len(sentences))

(sentences, next_words), (sentences_test, next_words_test) = shuffle_and_split_training_set(sentences, next_words)

model = Sequential()
model.add(Bidirectional(LSTM(256), input_shape=(sequence_length, len(words))))
if dropout > 0:
    model.add(Dropout(dropout))
model.add(Dense(len(words)))
model.add(Activation('softmax'))

print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

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
                    steps_per_epoch=int(len(sentences)/batch_size) + 1,
                    epochs=100,
                    callbacks=callbacks_list,
                    validation_data=generator(sentences_test, next_words_test, batch_size),
                    validation_steps=int(len(sentences_test)/batch_size) + 1)
