from SpeechDenoising.data_processing import DatasetHandler
from SpeechDenoising.models.handler import DeNoiser
from SpeechDenoising.training.losses import sp_dn_loss

import keras
import numpy as np

dataset_handler = DatasetHandler('../edi-data/clean_trainset_28spk_wav', '../edi-data/noisy_trainset_28spk_wav')
dataset_handler.shuffle_datalist()

from SpeechDenoising.training.losses import sp_dn_loss

dn_models = DeNoiser()
dn_models.denoiser.compile(optimizer=keras.optimizers.Adam(lr=1e-4, decay=1e-3),
                           loss=sp_dn_loss(fw_n_pow_weight=1,
                                           mae_weight=0,
                                           itakura_weight=0,
                                           sequences=False,
                                           log_noise=False))

batch_size = 2
buffer_snaps_per_audio = 2
hist = []
for epoch in range(100):

    losses = []
    for i, noisy, clean, name in dataset_handler.epoch_batches(batch_size=batch_size,
                                                                   random_buffer_snaps=buffer_snaps_per_audio):


        try:
            loss = dn_models.denoiser.train_on_batch(noisy, clean)
            losses.append(loss)
        except ValueError:
            print('ValueError while processing batch ' +str(i))

    hist.append(np.mean(losses))
    print('Loss at epoch ' + str(epoch) +': ' + str(hist[-1]))

dn_models.denoiser.save_weights('saved_weights.h5')
