import os
import pandas as pd
import numpy as np

dir_results = './Resultados/'


def analise_resultados():

    cenarios = {}

    for pasta in sorted(os.listdir(dir_results)):

        caminho = os.path.join(dir_results, pasta)

        if os.path.isdir(caminho):

            for result in sorted(os.listdir(caminho)):

                if result.endswith('.csv'):

                    arquivo = os.path.join(caminho, result)

                    df = pd.read_csv(arquivo)

                    cenario = result.split('_execucao')[0]

                    accuracy = df[df['Unnamed: 0'] ==
                                  'accuracy']['precision'].values[0]

                    macro_avg = df[df['Unnamed: 0'] == 'macro avg']

                    precision_macro = macro_avg['precision'].values[0]
                    recall_macro = macro_avg['recall'].values[0]
                    f1_macro = macro_avg['f1-score'].values[0]

                    if cenario not in cenarios:
                        cenarios[cenario] = []

                    cenarios[cenario].append({
                        'accuracy': accuracy,
                        'precision_macro': precision_macro,
                        'recall_macro': recall_macro,
                        'f1_macro': f1_macro
                    })

    for cenario, valores in cenarios.items():

        df_resultados = pd.DataFrame(valores)

        accuracy_media = df_resultados['accuracy'].mean() * 100
        precision_media = df_resultados['precision_macro'].mean() * 100
        recall_media = df_resultados['recall_macro'].mean() * 100
        f1_media = df_resultados['f1_macro'].mean() * 100

        accuracy_std = df_resultados['accuracy'].std() * 100
        precision_std = df_resultados['precision_macro'].std() * 100
        recall_std = df_resultados['recall_macro'].std() * 100
        f1_std = df_resultados['f1_macro'].std() * 100

        print(f'===== {cenario} =====')

        print(f'Accuracy média: {accuracy_media:.2f}%')
        print(f'Accuracy desvio padrão: {accuracy_std:.2f}%')

        print(f'Precision macro média: {precision_media:.2f}%')
        print(f'Precision macro desvio padrão: {precision_std:.2f}%')

        print(f'Recall macro média: {recall_media:.2f}%')
        print(f'Recall macro desvio padrão: {recall_std:.2f}%')

        print(f'F1-score macro média: {f1_media:.2f}%')
        print(f'F1-score macro desvio padrão: {f1_std:.2f}%')

        print()


analise_resultados()
