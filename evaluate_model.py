import pandas as pd
import os
import librosa
import librosa.display

import pysepm

from SpeechDenoising.models.handler import DeNoiser
from SpeechDenoising.data_processing import DatasetHandler
from SpeechDenoising.evaluation import evaluate_audio


def get_evaluation_audios(clean_dir='../edi-data/clean_trainset_28spk_wav',
                          noisy_dir='../edi-data/noisy_trainset_28spk_wav',
                          out_dir='../evaluation_audios'):

    dn_models = DeNoiser()
    dataset_handler = DatasetHandler(clean_dir, noisy_dir)

    # Process audios
    for i, noisy_audio, clean_audio, name in dataset_handler.epoch_batches(batch_size=1, random_buffer_snaps=False):

        evaluate_audio(dn_models.time_dist_denoiser,
                       clean_audio, noisy_audio, name[0], max_seg=10,
                       out_dir=out_dir)

# Compute metrics on the processed audios
def compute_metrics_per_audio(audio_dir='../evaluation_audios', out_file='../evaluation_metrics.csv'):

    f = open(out_file, 'w')
    f.write(','.join(['filename',
                      'n_snr', 'dn_snr', 'diff_snr',
                      'n_fwsnr', 'dn_fwsnr', 'diff_fwsnr',
                      'n_Csig', 'dn_Csig', 'diff_Csig',
                      'n_Cbak', 'dn_Cbak', 'diff_Cbak',
                      'n_Covl', 'dn_Covl', 'diff_Covl']) + '\n')

    for file in os.listdir(os.path.join(audio_dir, 'clean')):

            c, fs = librosa.core.load(os.path.join(audio_dir, 'clean', file))
            n, fs = librosa.core.load(os.path.join(audio_dir, 'noisy', file))
            dn, fs = librosa.core.load(os.path.join(audio_dir, 'denoised', file))

            metrics = []

            n_snr = pysepm.SNRseg(c, n, fs)
            dn_snr = pysepm.SNRseg(c, dn, fs)
            metrics += [n_snr,  dn_snr,  dn_snr - n_snr]

            n_fwsnr = pysepm.fwSNRseg(c, n, fs)
            dn_fwsnr = pysepm.fwSNRseg(c, dn, fs)
            metrics += [n_fwsnr,  dn_fwsnr,  dn_fwsnr - n_fwsnr]

            c_16khz = librosa.resample(c, fs, 16000)
            n_16khz = librosa.resample(n, fs, 16000)
            dn_16khz = librosa.resample(dn, fs, 16000)

            n_Csig, n_Cbak, n_Covl = pysepm.composite(c_16khz, n_16khz, 16000)
            dn_Csig, dn_Cbak, dn_Covl = pysepm.composite(c_16khz, dn_16khz, 16000)
            metrics += [n_Csig, dn_Csig, dn_Csig - n_Csig,
                        n_Cbak, dn_Cbak, dn_Cbak - n_Cbak,
                        n_Covl,  dn_Covl, dn_Covl - n_Covl]

            metrics = [str(round(m, 4)) for m in metrics]

            f.write(file.replace('.wav','') + ',' + ','.join(metrics) +'\n')

    f.close()


# Compute global metrics

def compute_global_metrics(metrics_per_audio_file='../evaluation_metrics.csv',
                           log_testset_path='../edi-data/logfiles/log_testset.txt'):

    df = pd.read_csv(metrics_per_audio_file)
    df = df.set_index('filename')

    df_key = pd.read_csv(log_testset_path, delim_whitespace=True, names=['filename', 'noisetype', 'snr'])
    df_key = df_key.set_index('filename')

    df = pd.concat([df, df_key], axis=1, sort=True)
    df = df.dropna()

    stat_series = []
    stat_columns = []
    for snr in df.snr.unique():

        this_df_range = df.loc[(df.snr == snr)]

        stat = this_df_range.mean().round(2).map(str) + ' (' + this_df_range.std().round(2).map(str) + ')'
        stat_series.append(stat)
        stat_columns += ['n_type' +' ' + str(snr) + " Mean (std)"]

    stats = pd.concat(stat_series, axis=1)
    stats.columns = stat_columns

    return stats


if __name__ == "__main__":
    # execute only if run as a script

    get_evaluation_audios()
    compute_metrics_per_audio()
    stats = compute_global_metrics()

    print(stats.to_string())
