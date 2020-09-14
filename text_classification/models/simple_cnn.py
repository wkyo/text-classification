# coding: utf-8
from keras import Sequential
from keras import layers


def build_simple_cnn_model(vocabulary_size=None,
                           n_classes=None,
                           embedding_size=300,
                           max_len=None,
                           hidden_size=128,
                           name=None):
    """Generate simple CNN model for text classification
    """
    assert vocabulary_size is not None
    assert n_classes is not None

    model = Sequential([
        layers.Embedding(vocabulary_size, embedding_size),
        layers.Conv1D(128, 7),
        layers.Conv1D(128, 7),
        layers.GlobalMaxPool1D(),
        layers.Dense(n_classes, activation='softmax')
    ], name=name)
    return model


EXPORTS = {
    'build': build_simple_cnn_model,
    'backend': 'keras',
    'name': 'SimpleCNN'
}
