import librosa as lb

import matplotlib.pyplot as plt

import numpy as np

import os

kick_path = './Dataset/archive/kick/'
snare_path = './Dataset/archive/snare/'
toms_path = './Dataset/archive/toms/'


taxa_amostragem = []
duracao = []

for audio in os.listdir(kick_path):
    if audio.endswith('.wav'):
        y, sr = lb.load(os.path.join(kick_path, audio), sr=None)
        taxa_amostragem.append(sr)
        duracao.append(lb.get_duration(y=y, sr=sr))
        
for audio in os.listdir(snare_path):
    if audio.endswith('.wav'):
        y, sr = lb.load(os.path.join(snare_path, audio), sr=None)
        taxa_amostragem.append(sr)
        duracao.append(lb.get_duration(y=y, sr=sr))
        
for audio in os.listdir(toms_path):
    if audio.endswith('.wav'):
        y, sr = lb.load(os.path.join(toms_path, audio), sr=None)
        taxa_amostragem.append(sr)
        duracao.append(lb.get_duration(y=y, sr=sr))
        
print(np.mean(duracao))

print("Média da taxa de amostragem: \n", np.mean(taxa_amostragem))
print("\nTaxas de amostragem: \n", taxa_amostragem)
print("\nMédia da duração: \n", duracao)
        