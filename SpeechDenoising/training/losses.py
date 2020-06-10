import tensorflow as tf
import keras.backend as K
from SpeechDenoising.config import BufferConfig


def join_sequence(x, buffer_config=BufferConfig):
    """Join a sequence of overlapping windowed segments into a single tensor."""
    x = x[:, :,  -buffer_config.OUTPUT_FRAME_LENGTH - buffer_config.TRAIL_SAMPLES: -buffer_config.TRAIL_SAMPLES]
    x_shape = K.shape(x)
    len_ = x_shape[1] * x_shape[2]
    x = K.reshape(x, (1, len_, 1))

    return x


def itakura_saito(ref, deg, stft_frame_length=512-1, stft_frame_step=64, fft_length=512-1):
    """Compute the Itakura-Saito distance between the spectrum of two tensors.

    Attributes
    ----------
    stft_frame_length --
    stft_frame_step --
    fft_length --

    """
    ref = K.reshape(ref, (-1, ))
    deg = K.reshape(deg, (-1, ))

    deg_s = tf.contrib.signal.stft(deg, frame_length=stft_frame_length, frame_step=stft_frame_step,
                                   fft_length=fft_length)
    deg_s = tf.abs(deg_s)

    ref_s = tf.contrib.signal.stft(ref, frame_length=stft_frame_length, frame_step=stft_frame_step,
                                   fft_length=fft_length)
    ref_s = tf.abs(ref_s)

    r = ref_s / deg_s
    q = r - K.log(r) - 1

    q = K.clip(q, -1000, 1000)

    distance = K.mean(K.sum(q, -1))

    return distance


def fw_seg_noise_power(ref, deg, weight_coef=0.2,  weighted_snr=True, log_noise=True,
                       stft_frame_length=1024-1, stft_frame_step=64, fft_length=1024-1):
    """Compute the frequency-weighted segmental SNR between two tensors.


       Attributes
       ----------
       weight_coef --
       weighted_snr --
       log_noise --
       stft_frame_length --
       stft_frame_step --
       fft_length --

    """
    ref = K.reshape(ref, (-1, ))
    deg = K.reshape(deg, (-1, ))

    deg_s = tf.contrib.signal.stft(deg, frame_length=stft_frame_length, frame_step=stft_frame_step,
                                   fft_length=fft_length)
    deg_s = tf.abs(deg_s)

    ref_s = tf.contrib.signal.stft(ref, frame_length=stft_frame_length, frame_step=stft_frame_step,
                                   fft_length=fft_length)
    ref_s = tf.abs(ref_s)

    dif_s = (ref_s - deg_s) ** 2

    if log_noise:

        numerator = tf.log(dif_s)
        denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
        dif_s = numerator / denominator

    if weighted_snr:

        weight = ref_s ** weight_coef
        dif = weight * dif_s
        snr_per_ts = K.sum(dif, axis=-1) / K.sum(weight, axis=-1)

    else:

        snr_per_ts = K.mean(dif_s, axis=-1)

    snr = K.mean(snr_per_ts, axis=-1)

    return snr


def sp_dn_loss(fw_n_pow_weight=-1, fw_n_pow_weight_coef=0.2, mae_weight=1, itakura_weight=-1, sequences=False, weighted_snr=True, log_noise=True,
               buffer_config=BufferConfig):
    """Build a cost function based on mean absolute error, fwSNRseg or IS distance. Within each audio segment, only the
       samples used to reconstruct the output audio are considered in the function.

       Attributes
       ----------
       fw_n_pow_weight
          Weight in the total loss of the frequency weighted segmental SNR.
       mae_weight
          Weight in the total loss of the mean absolute error.
       itakura_weight
          Weight in the total loss of the Itakura-Saito distance between spectra.
       sequences
          True if training a time distributed model, using several segments of each audio.
          If training on single segments, set to False.
       weighted_snr
          True to use SNR frequency weighting (default). Only if n_pow_weight > 0.
       log_noise
          True to consider logarithmic noise power (defaults to False). Only if n_pow_weight > 0.
    """

    def _loss(y_true, y_pred):

        seg_n_p = 0
        mae = 0
        itakura = 0

        if sequences:
            y_true = join_sequence(y_true)
            y_pred = join_sequence(y_pred)

        else:
            y_true = y_true[:, -buffer_config['OUTPUT_FRAME_LENGTH'] - buffer_config['TRAIL_SAMPLES']: -buffer_config['TRAIL_SAMPLES'], :]
            y_pred = y_pred[:, -buffer_config['OUTPUT_FRAME_LENGTH'] - buffer_config['TRAIL_SAMPLES']: -buffer_config['TRAIL_SAMPLES'], :]

        if mae_weight > 0:
            mae = K.squeeze(K.mean(K.mean(K.abs(y_pred - y_true), axis=-2), 0), -1)

        if fw_n_pow_weight > 0:
            seg_n_p = fw_seg_noise_power(y_true, y_pred, weighted_snr=weighted_snr, log_noise=log_noise, weight_coef=fw_n_pow_weight_coef)

        if itakura_weight > 0:
            itakura = itakura_saito(y_true, y_pred)

        return fw_n_pow_weight * seg_n_p + mae_weight * mae + itakura_weight * itakura

    return _loss
