import keras.backend as K
from keras.layers import *


def DilatedBlock(x, dilation, num_filters_1, num_filters_2):

    """Get a dilated convolutional block (TO-DO cite WaveNet)."""

    x1 = Conv1D(num_filters_1, 2, padding='causal',
                kernel_initializer='random_uniform', dilation_rate=dilation)(x)
    x2 = Conv1D(num_filters_1, 2, padding='causal',
                kernel_initializer='random_uniform', dilation_rate=dilation)(x)

    x1 = BatchNormalization()(x1)
    x2 = BatchNormalization()(x2)

    x1 = Activation('tanh')(x1)
    x2 = Activation('sigmoid')(x2)

    x_skip = multiply([x1, x2])
    x_skip = Conv1D(num_filters_2, 1, padding='same', kernel_initializer='random_uniform',
                    activation='tanh')(x_skip)

    x_res = add([x, x_skip])
    x_res = Conv1D(num_filters_2, 1, padding='same', kernel_initializer='random_uniform',
                   activation='tanh')(x_res)

    return x_skip, x_res


def Encoder(x, config):
    """Get layers for an encoder block."""
    for layer_spec in config['ENCODER_SPECS']:

        x = Conv1D(layer_spec['n_filt'],
                   layer_spec['filt_len'],
                   padding='same',
                   kernel_initializer='random_uniform',
                   kernel_regularizer=layer_spec['reg'],
                   activation=layer_spec['activation'],
                   strides=layer_spec['strides'])(x)

    encoded = x

    return encoded


def Conv1DTranspose(input_tensor, kernel_size, filters,  strides=4, padding='same'):
    """Transposed one-dimensional convolution."""
    _x = Lambda(lambda _x: K.expand_dims(_x, axis=2))(input_tensor)
    _x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(_x)
    _x = Lambda(lambda _x: K.squeeze(_x, axis=2))(_x)

    return _x


def Decoder(x, config):
    """Get layers for a decoder block."""
    for i, layer_spec in enumerate(config['DECODER_SPECS']['transpose']):

        if config['ENCODER_SPECS'][i]['strides'] > 1:
            x = Conv1DTranspose(x, layer_spec['filt_len'], layer_spec['n_filt'],
                                strides=config['ENCODER_SPECS'][i]['strides'], padding='same')

        x = Conv1D(layer_spec['n_filt'], layer_spec['filt_len'], padding='same',
                   kernel_initializer='random_uniform', kernel_regularizer=layer_spec['reg'],
                   activation=layer_spec['activation'], strides=layer_spec['strides'])(x)

    for i, layer_spec in enumerate(config['DECODER_SPECS']['final']):
        x = Conv1D(layer_spec['n_filt'], layer_spec['filt_len'], padding='same')(x)
        if 'activation' in layer_spec:
            x = Activation(layer_spec['activation'])(x)

    x = Conv1D(1, 1, padding='same', kernel_initializer='random_uniform')(x)
    x = Activation('tanh')(x)

    return x



