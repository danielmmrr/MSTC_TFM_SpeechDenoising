import numpy as np
import librosa
from SpeechDenoising.config import BufferConfig
from SpeechDenoising.data_processing import join_buffered
import os

__all__ = ['evaluate_audio']


def join_buffered(audio_segments):
    """Join a sequence of overlapping windowed segments into a single array."""

    audio_segments = audio_segments[:, :, -BufferConfig['OUTPUT_FRAME_LENGTH'] - BufferConfig['TRAIL_SAMPLES'] :
                                    - BufferConfig['TRAIL_SAMPLES'], :]
    au_shape = np.shape(audio_segments)

    len_ = au_shape[1] * au_shape[2]
    recovered = np.reshape(audio_segments, (len_,))

    return recovered


def evaluate_audio(td_model, clean_audio, noisy_audio, fn, max_seg=-1, out_dir='.'):

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for t in ('clean', 'noisy', 'denoised'):
        if not os.path.exists(os.path.join(out_dir, t)):
            os.mkdir(os.path.join(out_dir, t))

    clean_out_dir = os.path.join(out_dir, 'clean')
    noisy_out_dir = os.path.join(out_dir, 'noisy')
    denoised_out_dir = os.path.join(out_dir, 'denoised')

    clean_audio_fn = clean_out_dir + '/' + fn
    noisy_audio_fn = noisy_out_dir + '/' + fn
    denoised_audio_fn = denoised_out_dir + '/' + fn

    noisy_ = noisy_audio[:, :max_seg, :, :]
    clean_ = clean_audio[:, :max_seg, :, :]

    noisy_ = np.transpose(noisy_, (0, 1, 3, 2))
    clean_ = np.transpose(clean_, (0, 1, 3, 2))

    denoised = td_model.predict(noisy_)

    denoised = join_buffered(denoised)
    noisy = join_buffered(noisy_)
    clean_ref = join_buffered(clean_)

    librosa.output.write_wav(denoised_audio_fn, denoised, sr=48000)
    librosa.output.write_wav(noisy_audio_fn, noisy, sr=48000)
    librosa.output.write_wav(clean_audio_fn, clean_ref, sr=48000)

    return denoised_audio_fn, noisy_audio_fn, clean_audio_fn
