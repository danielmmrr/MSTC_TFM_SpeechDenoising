from random import shuffle, seed, randint
import os
import numpy as np
from .audio_tools import read_wave, through_buffer, process_audio, random_buffer_marks
from SpeechDenoising.config import AudioConfig, BufferConfig

__all__ = ['DatasetHandler']


class DatasetHandler:
    """
    Class used to handle speech datasets

    Uses configuration from configs.py extensively.
    ...

    Attributes
    ----------
    clean_dir_path : str

    noisy_dir_path : str


    Methods
    -------
    shuffle_datalist
       Shuffles the data for training
    epoch_batches(batch_size=1, buffer=False, vad_masking=False)
       Iterates through the dataset returning batches of processed clean and noisy audio for training
    batch_through_buffer

    get_single_audio

    get_audios

    """

    def __init__(self, clean_dir_path, noisy_dir_path, audio_config=AudioConfig, buffer_config=BufferConfig):

        self.clean_dir_path = clean_dir_path
        self.noisy_dir_path = noisy_dir_path

        self.audio_list = os.listdir(clean_dir_path)

        self.AudioConfig = audio_config
        self.BufferConfig = buffer_config

    def shuffle_datalist(self):

        seed(1)
        shuffle(self.audio_list)

    def epoch_batches(self, batch_size=1, random_buffer_snaps=True):

        curr_batch = 0
        while (curr_batch * batch_size) <= (len(self.audio_list )):

            first_index = curr_batch * batch_size
            batch_filenames = self.audio_list[first_index:first_index + batch_size]

            clean_batch = []; noisy_batch = []; names = batch_filenames
            for fn in batch_filenames:

                c_pcm_data, c_float_data, c_sample_rate = read_wave(os.path.join(self.clean_dir_path, fn))
                n_float_data = read_wave(os.path.join(self.noisy_dir_path, fn))[1]

                c_float_data = process_audio(c_float_data, sr=c_sample_rate, config=self.AudioConfig)
                n_float_data = process_audio(n_float_data, sr=c_sample_rate, config=self.AudioConfig)

                if random_buffer_snaps:

                    for sn in range(random_buffer_snaps):

                        ind = randint(self.BufferConfig['BUFFER_LENGTH'], len(c_float_data))
                        snap_start, snap_end = random_buffer_marks(self.BufferConfig['BUFFER_LENGTH'], len(c_float_data), ind)

                        clean_batch.append(c_float_data[snap_start:snap_end])
                        noisy_batch.append(n_float_data[snap_start:snap_end])
                else:

                    clean_batch.append(through_buffer(c_float_data, max_receptive_field=self.BufferConfig['MAX_RECEPTIVE_FIELD'],
                                       frame_len=self.BufferConfig['OUTPUT_FRAME_LENGTH']))

                    noisy_batch.append(through_buffer(n_float_data, max_receptive_field=self.BufferConfig['MAX_RECEPTIVE_FIELD'],
                                       frame_len=self.BufferConfig['OUTPUT_FRAME_LENGTH']))

            noisy_batch = np.expand_dims(noisy_batch, axis=2)
            clean_batch = np.expand_dims(clean_batch, axis=2)

            yield curr_batch, noisy_batch, clean_batch, names
            curr_batch += 1

    def batch_through_buffer(self, n_audios, c_audios, masks):

        b_n_audios = []
        b_c_audios = []
        b_masks = []
        for c_audio, n_audio, masks in zip(c_audios, n_audios, masks):

            b_n_audios.append(through_buffer(n_audio,
                                             max_receptive_field=self.BufferConfig['MAX_RECEPTIVE_FIELD'],
                                             frame_len=self.BufferConfig['OUTPUT_FRAME_LENGTH']))
            b_c_audios.append(through_buffer(c_audio,
                                             max_receptive_field=self.BufferConfig['MAX_RECEPTIVE_FIELD'],
                                             frame_len=self.BufferConfig['OUTPUT_FRAME_LENGTH']))
            b_masks.append(through_buffer(masks,
                                          max_receptive_field=self.BufferConfig['MAX_RECEPTIVE_FIELD'],
                                          frame_len=self.BufferConfig['OUTPUT_FRAME_LENGTH']))

        b_n_audios = np.asarray(b_n_audios)
        b_c_audios = np.asarray(b_c_audios)
        b_masks = np.asarray(b_masks)

        return b_n_audios, b_c_audios, b_masks

    def get_single_audio(self, index):

        fn = self.audio_list[index]

        c_pcm_data, c_float_data, c_sample_rate = read_wave(os.path.join(self.clean_dir_path, fn))
        n_float_data = read_wave(os.path.join(self.noisy_dir_path, fn))[1]

        c_audio = process_audio(c_float_data, sr=c_sample_rate, config=self.AudioConfig)
        n_audio = process_audio(n_float_data, sr=c_sample_rate, config=self.AudioConfig)

        noisy_buffered = through_buffer(n_audio, max_receptive_field=self.BufferConfig['MAX_RECEPTIVE_FIELD'],
                                        frame_len=self.BufferConfig['OUTPUT_FRAME_LENGTH'])
        clean_buffered = through_buffer(c_audio, max_receptive_field=self.BufferConfig['MAX_RECEPTIVE_FIELD'],
                                        frame_len=self.BufferConfig['OUTPUT_FRAME_LENGTH'])

        return dict(noisy=n_audio, noisy_buffered=noisy_buffered,
                    clean=c_audio, clean_buffered=clean_buffered)
