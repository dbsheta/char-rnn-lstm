import sys
import time

import numpy as np
from keras.layers import LSTM, Dense, TimeDistributed, Activation, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop

# Maximum length of continuous sequence
MAXLEN = 40

# One Character at a time i.e [a,b,c] -> [b,c,d] .....predict the next character
STEP = 1
LEN_CHARSET = 0


# Build the recurrent neural network with LSTM
def build_model(no_of_chars):
    model = Sequential()
    model.add(LSTM(512, input_dim=no_of_chars, return_sequences=True))
    model.add(TimeDistributed(Dense(no_of_chars)))
    model.add(Activation('softmax'))
    model.compile(optimizer=RMSprop(0.01), loss='categorical_crossentropy')
    return model


# Data Handling ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def process_data():
    with open('poems.txt') as corpus:
        print("Reading data")
        text_data = corpus.read().lower()
        print(f'Length of dataset: {len(text_data)}')
    character_set = sorted(list(set(text_data)))
    char_to_index = {c: i for i, c in enumerate(character_set)}
    index_to_char = {i: c for c, i in char_to_index.items()}
    l = len(character_set)
    # Generate sentences having length MAXLEN and the expected prediction
    sentences = []
    pred_sentences = []
    for i in range(0, len(text_data) - MAXLEN + 1, STEP):
        sentences.append(text_data[i: i + MAXLEN])
        pred_sentences.append(text_data[i + STEP: i + STEP + MAXLEN])
    del text_data
    return sentences, pred_sentences, char_to_index, index_to_char, l


def vectorize_input_data(sentences, char_map, vec_len=MAXLEN):
    X = np.zeros((len(sentences), vec_len, LEN_CHARSET), dtype=np.bool)
    # transform into one-hot
    for i, sentence in enumerate(sentences):
        for j, char in enumerate(sentence):
            X[i, j, char_map[char]] = 1
    return X


def vectorize_output_data(sentences, pred_sentences, char_map):
    Y = np.zeros((len(sentences), MAXLEN, LEN_CHARSET), dtype=np.bool)
    # transform into one-hot
    for i, sentence in enumerate(pred_sentences):
        for j, char in enumerate(sentence):
            Y[i, j, char_map[char]] = 1
    return Y


# Training Neural Network~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def train(model, x_train, y_train, no_epochs):
    start = time.time()
    for i in range(no_epochs):
        print('-' * 10)
        print("Iteration", i)
        history = model.fit(x_train, y_train, batch_size=256, nb_epoch=1)

        # Save intermediate weights after 3 iterations
        if i % 3 == 0:
            model.save_weights(f'rnn{i}.h5', overwrite=True)
            print("Saved weights backup")
        print(history.history['loss'])
    print(f'Training completed in {time.time() - start}')


# Making Neural Network Generate Text based on some input ~~~~~~~~~~~~~~~~~~
def generate_text(model, seed_string, len_generated_text, char_map, index_map):
    print(f'{seed_string}', end='\n\n')
    for i in range(len_generated_text):
        x_test = vectorize_input_data([seed_string], char_map, len(seed_string))
        pred_result = model.predict(x_test, verbose=2)[0]
        next_char_i = np.argmax(pred_result[len(seed_string) - 1])
        next_char = index_map[next_char_i]
        seed_string += next_char
        sys.stdout.write(next_char)
        sys.stdout.flush()


if __name__ == '__main__':
    data = process_data()
    LEN_CHARSET = data[-1]
    print("1. Train New")
    print("2. Train Existing")
    print("3. Generate Some Text")
    print("4. Summary")
    choice = int(input())
    if choice == 1:
        rnn = build_model(LEN_CHARSET)
        x = vectorize_input_data(data[0], data[2])
        y = vectorize_output_data(*data[:3])
        train(rnn, x, y, 1)
        rnn.save("rnn_model.h5")
    elif choice == 2:
        rnn = load_model('rnn_model.h5')
        x = vectorize_input_data(data[0], data[2])
        y = vectorize_output_data(*data[:3])
        train(rnn, x, y, 5)
    elif choice == 3:
        rnn = load_model("rnn_model.h5")
        print("Enter name for the poem: ", end='')
        seed = input().lower()
        print("No of characters in poem:" ,end='')
        no_chars = int(input())
        print(generate_text(rnn, seed, no_chars, data[2], data[3]))
    else:
        rnn = load_model("rnn_model.h5")
        rnn.summary()
