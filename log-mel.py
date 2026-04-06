import librosa as lb

import matplotlib.pyplot as plt

import numpy as np

import os


def plot_melspectrogram(name_step):
    classes = ['kick', 'snare', 'toms']
    
    for classe in classes:
        path = './Dataset/' + name_step + '/' + classe + '/'
        for audio in os.listdir(path):
            if audio.endswith('.wav'):
                y, sr = lb.load(os.path.join(path, audio), sr=None)
                melspec = lb.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=None)
                fig, ax = plt.subplots()
                melspec_log = lb.power_to_db(melspec, ref=np.max)
                img = lb.display.specshow(melspec_log, x_axis='time', y_axis='mel', sr=sr, fmax=None, ax=ax)
                fig.colorbar(img, ax=ax, format='%+2.0f dB')
                ax.set(title='Mel-frequency spectrogram')
                plt.savefig(f'./Dataset/' + name_step + '/' + classe + '/' + os.path.splitext(os.path.basename(audio))[0] + '_log-mel.png')
                plt.close(fig)
        

plot_melspectrogram('train')
plot_melspectrogram('validation')
plot_melspectrogram('test')



