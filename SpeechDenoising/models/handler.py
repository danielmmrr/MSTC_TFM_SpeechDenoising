from SpeechDenoising.config import BufferConfig, DeepModelConfig
from SpeechDenoising.data_processing.audio_tools import join_buffered

from .create_models import build_encoder, build_masker, build_decoder, build_denoiser, time_distributed, build_autoencoder
import numpy as np
import os

weights_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_weights')


class DeNoiser:

    def __init__(self, model_config=DeepModelConfig, buffer_config=BufferConfig, preset=None):

        if preset == 'mae':
            self.buffer_config = BufferConfig
            self.model_config = DeepModelConfig
            self.build_denoiser()
            self.update_denoiser_weights(os.path.join(weights_dir, 'saved_weights_mae-2.h5'))

        elif preset == 'fwsnr':
            self.buffer_config = BufferConfig
            self.model_config = DeepModelConfig
            self.build_denoiser()
            self.update_denoiser_weights(os.path.join(weights_dir, 'saved_weights_fwsnrseg.h5'))

        else:
            self.buffer_config = buffer_config
            self.model_config = model_config
            self.build_denoiser()

    def build_denoiser(self):

        self.encoder = build_encoder(buffer_length=self.buffer_config['BUFFER_LENGTH'], config=self.model_config)
        self.masker = build_masker(self.encoder.output_shape[1:], config=self.model_config)

        self.decoder = build_decoder(self.encoder.output_shape[1:], config=self.model_config)

        self.denoiser = build_denoiser(self.encoder, self.decoder, self.masker)
        self.time_dist_denoiser = time_distributed(self.denoiser, self.buffer_config['BUFFER_LENGTH'])

        #self.autoencoder = build_autoencoder(self.encoder, self.decoder)


    def update_denoiser(self, denoiser):

        self.denoiser = denoiser
        self.time_dist_denoiser = time_distributed(self.denoiser, self.buffer_config['BUFFER_LENGTH'])

    def update_denoiser_weights(self, denoiser_weights_path):

        self.denoiser.load_weights(denoiser_weights_path)
        self.time_dist_denoiser = time_distributed(self.denoiser, self.buffer_config['BUFFER_LENGTH'])

    def denoise(self, audio):

        noisy_buffered = np.expand_dims([audio], axis=-1)
        denoised_buffered = self.time_dist_denoiser.predict(noisy_buffered)
        denoised_audio = join_buffered(denoised_buffered, self.buffer_config)

        return denoised_audio
