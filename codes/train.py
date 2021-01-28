#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/28 15:45
@File:          train.py
'''

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import *
from keras.optimizers import Adam
from keras import Model

from PositionEmbedding import SinusoidalPositionEmbedding
from Attention import Attention

max_words = 20000
maxlen = 100
embed_dim = 64
batch_size = 1

(x_train, y_train), (x_val, y_val) = imdb.load_data(num_words=max_words)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = sequence.pad_sequences(x_val, maxlen=maxlen)

text_input = Input(shape=(maxlen, ), dtype='int32')
x = Embedding(max_words, embed_dim)(text_input)
x = SinusoidalPositionEmbedding()(x)
x = Attention(use_scale=True)([x, x])
x = GlobalAveragePooling1D()(x)
out = Dense(1, activation='sigmoid')(x)

model = Model(text_input, out)
model.compile(Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=10, validation_data=(x_val, y_val))