import librosa as lb
import matplotlib.pyplot as plt
import numpy as np
import os

dataset_dir = './Dataset/archive/'

def audio_info(path):
    taxa_amostragem = []
    duracao = []
    classes = ['kick', 'snare', 'toms']
    for i in classes:
        for audio in os.listdir(path + i + '/'):
            if audio.endswith('.wav'):
                y, sr = lb.load(os.path.join(path, i + '/', audio), sr=None)
                taxa_amostragem.append(sr)
                duracao.append(lb.get_duration(y=y, sr=sr))
    return taxa_amostragem, duracao

taxa_amostragem, duracao = audio_info(dataset_dir)
        
print("Média duração: ", np.mean(duracao))
print("Mínima duração: ", np.min(duracao))
print("Máxima duração: ", np.max(duracao))
print("Quantidade de arquivos: ", len(duracao))
print("\nDuracão do áudio em segundos: \n", duracao)


print("\nMédia da taxa de amostragem: ", np.mean(taxa_amostragem))
print("Mínima taxa de amostragem: ", np.min(taxa_amostragem))
print("Máxima taxa de amostragem: ", np.max(taxa_amostragem))
print("Quantidade de arquivos: ", len(taxa_amostragem))
print("\nTaxas de amostragem: \n", taxa_amostragem)
        