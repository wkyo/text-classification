# coding: utf-8
from keras import Model
from keras import layers


def build_text_rcnn_model(vocabulary_size=None,
                          n_classes=None,
                          embedding_size=300,
                          max_len=None,
                          hidden_size=128,
                          conv_kernels=None,
                          name=None):
    """Generate Text-RCNN model for text classification

    .. Lai S, Xu L, Liu K, et al. Recurrent convolutional neural networks for text classification[C]//Twenty-ninth AAAI conference on artificial intelligence. 2015.
    """
    assert vocabulary_size is not None
    assert n_classes is not None

    if not conv_kernels:
        conv_kernels = [2, 3, 4]

    input_ = layers.Input(shape=(max_len,))
    embedding = layers.Embedding(
        vocabulary_size, embedding_size)(input_)

    x = embedding
    # merge c_l, c_r and e_i, which e_i is the output of embedding, c_l and c_r is generated by Bi-RNN
    #       x_i = [c_l; c_r; e_i]
    x = layers.Bidirectional(layers.GRU(
        128, return_sequences=True), merge_mode='concat')(x)
    x = layers.Concatenate(axis=-1)([x, embedding])
    # apply linear transformation on each x_i, and activate with tanh
    #       y_i^{(2)} = tanh(W^{(2)x_i + b^{(2)}})
    x = layers.Dense(128, activation='tanh')(x)
    x = layers.GlobalMaxPool1D()(x)
    x = layers.Dense(n_classes, activation='softmax')(x)

    output = x

    model = Model(inputs=input_, outputs=output, name=name)
    return model


EXPORTS = {
    'build': build_text_rcnn_model,
    'backend': 'keras',
    'name': 'TextRCNN'
}
