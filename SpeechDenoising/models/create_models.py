from .building_blocks import *
from keras import Model
import keras

__all__ = ['build_encoder', 'build_decoder', 'build_masker',
           'build_autoencoder', 'build_denoiser']


def build_encoder(buffer_length, config):
    """Get encoder block."""
    inputs = Input(shape=(buffer_length, 1))
    encoded = Encoder(inputs, config)
    model = Model(inputs=inputs, outputs=encoded)

    return model


def build_decoder(encoder_output_shape, config):
    """Get decoder block."""
    inputs = Input(shape=encoder_output_shape)
    recovered = Decoder(inputs, config)
    model = Model(inputs=inputs, outputs=recovered)

    return model


def build_masker(encoder_output_shape, config):
    """Get masker block."""
    encoded = Input(shape=encoder_output_shape)

    x_skip, x_res = DilatedBlock(encoded,
                                 dilation=config['DILATIONS'][0],
                                 num_filters_1=config['MASKER_NUM_FILTERS'],
                                 num_filters_2=config['LATENT_REP_NUM_FILTERS'])
    total_skip = x_skip
    for dilation in config['DILATIONS'][1:]:
        x_skip, x_res = DilatedBlock(encoded,
                                 dilation=dilation,
                                 num_filters_1=config['MASKER_NUM_FILTERS'],
                                 num_filters_2=config['LATENT_REP_NUM_FILTERS'])
        total_skip = add([total_skip, x_skip])

    mask = Activation('sigmoid')(total_skip)

    x = multiply([encoded, mask])
    denoised = Activation('relu')(x)
    model = Model(inputs=encoded, outputs=denoised)

    return model


def build_autoencoder(encoder, decoder):
    """Create autoencoder from encoder and decoder blocks."""
    inputs = Input(shape=encoder.input_shape[1:])

    encoded = encoder(inputs)
    decoded = decoder(encoded)

    model = Model(inputs=inputs, outputs=decoded)

    return model


def build_denoiser(encoder, decoder, masker):
    """Create denoiser from encoder, decoder and masker blocks."""
    inputs = Input(shape=encoder.input_shape[1:])

    encoded = encoder(inputs)
    denoised = masker(encoded)
    decoded = decoder(denoised)

    model = Model(inputs=inputs, outputs=decoded)

    return model


def time_distributed(model, buff_len):
    """Create time distributed version of a denoiser model."""
    input_seq = keras.layers.Input(shape=(None, buff_len, 1))
    output_seq = keras.layers.TimeDistributed(model)(input_seq)
    td_model = keras.Model(inputs=input_seq, outputs=output_seq)

    return td_model
