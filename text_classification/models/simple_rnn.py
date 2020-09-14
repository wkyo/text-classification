# coding: utf-8
from keras import Sequential
from keras import layers


def build_simple_bi_rnn_model(vocabulary_size=None,
                              n_classes=None,
                              embedding_size=300,
                              max_len=None,
                              hidden_size=128,
                              name=None):
    """Generate simple Bi-RNN model for text classification
    """
    assert vocabulary_size is not None
    assert n_classes is not None

    model = Sequential([
        layers.Embedding(vocabulary_size, embedding_size),
        layers.Bidirectional(layers.GRU(hidden_size), merge_mode='concat'),
        # dense layer (shape [k, j]) of keras only works on the last axis of input (shape [batch_size, time_steps, k])
        #       [batch_size, time_steps, k] * [k, j] -> [batch_size, time_steps, j]
        # so, we don't need to use TimeDistributed layer on Dense layer
        layers.Dense(n_classes, activation='softmax')
    ], name=name)
    return model


EXPORTS = {
    'build': build_simple_bi_rnn_model,
    'backend': 'keras',
    'name': 'SimpleRNN'
}
