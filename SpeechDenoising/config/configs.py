from keras.regularizers import l2


BufferConfig = {

    'BUFFER_LENGTH': 24000,
    'MAX_RECEPTIVE_FIELD': 24000 - 8000,
    'OUTPUT_FRAME_LENGTH': 8000,
    'TRAIL_SAMPLES': 1000

}

DeepModelConfig = {

    'F_LEN': 48,
    'F_LEN_2': 24,
    'LATENT_REP_NUM_FILTERS': 90,
    'MASKER_NUM_FILTERS': 90,

    'ENCODER_SPECS': [dict(n_filt=90, filt_len=48, strides=1, reg=l2(0.01), activation='relu'),
                      dict(n_filt=90, filt_len=24, strides=1, reg=None, activation='relu')],

    'DECODER_SPECS': {'transpose': [dict(n_filt=90, filt_len=24, strides=1, reg=None, activation='relu'),
                                    dict(n_filt=90, filt_len=48, strides=1, reg=None, activation='relu')],
                      'final': [dict(n_filt=60, filt_len=6, activation='tanh')]
                      },

    'DILATIONS': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

}


class AudioConfig:

    NORMALIZE = True
    RESAMPLE_TO = None
    VAD_MASKING = False
    VAD_SENSIBILITY = 3


class TrainingConfig:

    PRINT_UPDATE_FREQ = 1
