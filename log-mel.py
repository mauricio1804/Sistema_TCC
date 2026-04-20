import librosa as lb

import matplotlib.pyplot as plt

import numpy as np

import os


train = lb.util.find_files('./Dataset/train/', ext=['wav'])
print(train)
exit()


def plot_melspectrogram(name_step):
    classes = ['kick', 'snare', 'toms']
    
    for classe in classes:
        path = './Dataset/' + name_step + '/' + classe + '/'
        for audio in os.listdir(path):
            if audio.endswith('.wav'):
                y, sr = lb.load(os.path.join(path, audio), sr=None)
                melspec = lb.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=None, n_fft=2048, hop_length=512)
                melspec_log = lb.power_to_db(melspec, ref=np.max)
                # plt.imsave(f'./Dataset/' + name_step + '/' + classe + '/' + os.path.splitext(os.path.basename(audio))[0] + '_log-mel.png', melspec_log)
plot_melspectrogram('validation')
plot_melspectrogram('test')
plot_melspectrogram('train')


