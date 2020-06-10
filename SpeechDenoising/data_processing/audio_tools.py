import numpy as np
import contextlib
import wave
import struct
from SpeechDenoising.config import BufferConfig

__all__ = ['join_buffered']


def process_audio(audio, sr, config):

    if config.NORMALIZE:
        audio = normalize_audio(audio)

    if config.RESAMPLE_TO:
        from librosa.core import resample
        audio = resample(audio, sr, config.RESAMPLE_TO)

    return audio


def random_buffer_marks(buffer_length, audio_length, random_int):

    assert random_int >= buffer_length
    assert random_int <= audio_length

    return random_int - buffer_length, random_int


def read_wave(path):
    """Read a raw audio file."""
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())

        float_data = struct.unpack("%ih" % (wf.getnframes() * wf.getnchannels()), pcm_data)
        float_data = [float(val) / pow(2, 15) for val in float_data]

        return pcm_data, float_data, sample_rate


def float2pcm(sig, dtype='int32'):
    """Convert audio signal to PCM."""
    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)  # +128


def normalize_audio(audio):
    """TO-DO. Normalize audio signal."""

    #audio = float2pcm(audio) / 2147483647
    #audio = audio.astype(np.float32)
    return np.asarray(audio)


def through_buffer(audio, max_receptive_field, frame_len):
    """Split audio into overlapping segments.

       Segments overlap only on the left-side samples.

       Attributes
       ----------
       max_receptive_field
          Distance (in samples) from first sample in the segment to the last sample present in the previous segment.
       frame_len
          Number of samples on the right side of the segment to be considered as output.
    """
    audio_in_buffers = []
    n = 0
    while max_receptive_field + (n+1)*frame_len < len(audio):

        buff = audio[n*frame_len: max_receptive_field + (n+1)*frame_len]
        audio_in_buffers.append(buff)
        n += 1

    return audio_in_buffers


def join_buffered(audio_segments, buffer_config=BufferConfig):
    """Join a sequence of overlapping windowed segments into a single array."""

    audio_segments = audio_segments[:, :, -buffer_config['OUTPUT_FRAME_LENGTH'] -buffer_config['TRAIL_SAMPLES']:
                                    -buffer_config['TRAIL_SAMPLES'], :]
    au_shape = np.shape(audio_segments)

    len_ = au_shape[1] * au_shape[2]
    recovered = np.reshape(audio_segments, (len_,))

    return recovered
