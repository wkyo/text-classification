# coding: utf-8
from keras import Model
from keras import layers


def build_text_cnn_model(vocabulary_size=None,
                         n_classes=None,
                         embedding_size=300,
                         max_len=None,
                         hidden_size=128,
                         conv_kernels=None,
                         name=None):
    """Generate Text-CNN model for text classification
    """
    assert vocabulary_size is not None
    assert n_classes is not None

    if not conv_kernels:
        conv_kernels = [2, 3, 4]

    input_ = layers.Input(shape=(max_len,))
    embedding = layers.Embedding(
        vocabulary_size, embedding_size)(input_)

    x = embedding

    multi_convs = []
    for kernel_size in conv_kernels:
        conv = layers.Conv1D(hidden_size, kernel_size, activation='relu')(x)
        conv = layers.GlobalMaxPool1D()(conv)
        multi_convs.append(conv)

    x = layers.Concatenate()(multi_convs)
    x = layers.Dense(n_classes, activation='softmax')(x)

    output = x

    model = Model(inputs=input_, outputs=output, name=name)
    return model


EXPORTS = {
    'build': build_text_cnn_model,
    'backend': 'keras',
    'name': 'TextCNN'
}
