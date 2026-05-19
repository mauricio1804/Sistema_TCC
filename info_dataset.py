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

taxa_amostragem = np.array(taxa_amostragem)
duracao = np.array(duracao)

verificacao_duracao = (np.allclose(duracao, 2.0) and len(duracao) == 120)

verificacao_taxa_amostragem = (np.all(taxa_amostragem == 44100) and len(taxa_amostragem) == 120)

if (verificacao_duracao):
    print("Todos os 120 arquivos de áudio têm a duração correta de 2 segundos.")
    print('valores únicos da duração: ', np.unique(duracao))
else:
    print("Nem todos os arquivos de áudio têm a duração correta de 2 segundos. Verifique os dados.")
    print("Média duração: ", np.mean(duracao))
    print("Mínima duração: ", np.min(duracao))
    print("Máxima duração: ", np.max(duracao))
    print('valores únicos da duração: ', np.unique(duracao))
    print("Quantidade de arquivos lidos: ", len(duracao))


if (verificacao_taxa_amostragem):
    print("Todos os 120 arquivos de áudio têm a taxa de amostragem correta de 44100 Hz.")
    print("Valores únicos da taxa de amostragem: ", np.unique(taxa_amostragem))
else:
    print("Nem todos os arquivos de áudio têm a taxa de amostragem correta de 44100 Hz. Verifique os dados.")
    print("\nMédia da taxa de amostragem: ", np.mean(taxa_amostragem))
    print("Mínima taxa de amostragem: ", np.min(taxa_amostragem))
    print("Máxima taxa de amostragem: ", np.max(taxa_amostragem))
    print("Valores únicos da taxa de amostragem: ", np.unique(taxa_amostragem))
    print("Quantidade de arquivos lidos: ", len(taxa_amostragem))
