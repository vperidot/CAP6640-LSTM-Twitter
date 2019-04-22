"""
Defines the LSTM Model

@author Ity Bahadur, Victoria Proetsch
"""

import keras
from keras import models
from keras.layers import LSTM, Dense, Embedding, Dropout, Input, Flatten
from keras import optimizers
from keras import metrics

def make_model(vocab_size=2500, seq_len=8, embedding_size=64, layers=3, channels=64, 
                sequences=False, dropout=0.25, batch_norm=False, stateful=False):
    """
    Creates a Keras Sequential model based on the given arguments.
    @return model, a compiled Keras model
    """
    
    # start the model
    model = models.Sequential()

    # embedding layer learns to transform index sequence to an n-dimensional embedding
    model.add(Embedding(vocab_size, embedding_size, input_length=seq_len))
    # each layer contributes to the "depth" of the model, i.e. the number of stacked LSTM layers
    for i in range(layers-1):
        # returns a sequence of vectors of dimension given by channels
        model.add(LSTM(channels, return_sequences=True)) 
        # Dropout regularizes the model by setting some weights to 0
        model.add(Dropout(rate=dropout))
    model.add(LSTM(channels, return_sequences=sequences)) # return either a sequence or last prediction
    model.add(Dropout(rate=dropout))
    # The dense layer transforms our output to the same vector space as our vocabulary (either words or chars)
    model.add(Dense(vocab_size, activation='softmax'))

    # Adam is a common choice proven to work in a variety of scenarios
    adam = optimizers.Adam(lr=1e-3)

    # Compile and return
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    return model

