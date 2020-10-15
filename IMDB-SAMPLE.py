import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd

print(tf.__version__)

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
word_inedx = imdb.get_word_index()
word_inedx = {k:(v+3) for k, v in word_inedx.items()}
word_inedx["<PAD>"] = 0
word_inedx["<START>"] = 1
word_inedx["<UNK>"] = 2
word_inedx["<UNUSED>"] = 3

reversed_word_index = dict([(value, key) for (key, value) in word_inedx.items()])

print(reversed_word_index)

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_inedx["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_inedx["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'], learning_rate=0.001)

x_val = train_data[:10000]
partial_x_train = train_data[10000:]


y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=300,
                    batch_size=64,
                    validation_data=(x_val, y_val),
                    verbose=1)
results = model.evaluate(test_data, test_labels)
print(results)