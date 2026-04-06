import sklearn as sk
import os
import shutil

dataset_dir = './Dataset/archive'

kicks_dir = os.listdir(os.path.join(dataset_dir, 'kick/'))
snare_dir = os.listdir(os.path.join(dataset_dir, 'snare/'))
toms_dir = os.listdir(os.path.join(dataset_dir, 'toms/'))

Arquivos_kicks = [f for f in kicks_dir if f.endswith('.wav')]
Arquivos_snare = [f for f in snare_dir if f.endswith('.wav')]
Arquivos_toms = [f for f in toms_dir if f.endswith('.wav')]

def kicks (arqui):
    k_train, K_temp = sk.model_selection.train_test_split(arqui, test_size=0.3, random_state=42)

    K_vali, K_test = sk.model_selection.train_test_split(K_temp, test_size=0.5, random_state=42)

    for arquivo in k_train:
        shutil.copy2(os.path.join(dataset_dir, 'kick/', arquivo), './Dataset/train/kick/')
        
    for arquivo in K_vali:
        shutil.copy2(os.path.join(dataset_dir, 'kick/', arquivo), './Dataset/validation/kick/')

    for arquivo in K_test:
        shutil.copy2(os.path.join(dataset_dir, 'kick/', arquivo), './Dataset/test/kick/')
        
def snare (arqui):
    S_train, S_temp = sk.model_selection.train_test_split(arqui, test_size=0.3, random_state=42)

    S_vali, S_test = sk.model_selection.train_test_split(S_temp, test_size=0.5, random_state=42)

    for arquivo in S_train:
        shutil.copy2(os.path.join(dataset_dir, 'snare/', arquivo), './Dataset/train/snare/')
        
    for arquivo in S_vali:
        shutil.copy2(os.path.join(dataset_dir, 'snare/', arquivo), './Dataset/validation/snare/')

    for arquivo in S_test:
        shutil.copy2(os.path.join(dataset_dir, 'snare/', arquivo), './Dataset/test/snare/')

def toms (arqui):
    T_train, T_temp = sk.model_selection.train_test_split(arqui, test_size=0.3, random_state=42)

    T_vali, T_test = sk.model_selection.train_test_split(T_temp, test_size=0.5, random_state=42)

    for arquivo in T_train:
        shutil.copy2(os.path.join(dataset_dir, 'toms/', arquivo), './Dataset/train/toms/')
        
    for arquivo in T_vali:
        shutil.copy2(os.path.join(dataset_dir, 'toms/', arquivo), './Dataset/validation/toms/')

    for arquivo in T_test:
        shutil.copy2(os.path.join(dataset_dir, 'toms/', arquivo), './Dataset/test/toms/')
        
kicks(Arquivos_kicks)
snare(Arquivos_snare)
toms(Arquivos_toms)