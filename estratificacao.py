import sklearn as sk
import os
import shutil

dataset_dir = './Dataset/archive'

classes = ['kick', 'snare', 'toms']

dir_train = './Dataset/train'
dir_validation = './Dataset/validation'
dir_test = './Dataset/test'


def list_wavs(path):
    return [f for f in os.listdir(path) if f.endswith('.wav')]


def count_wavs(path):
    return len(list_wavs(path))


def delete_files(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def verification(classe):
    if count_wavs(os.path.join(dir_train, classe)) == 0:
        return False
    if count_wavs(os.path.join(dir_validation, classe)) == 0:
        return False
    if count_wavs(os.path.join(dir_test, classe)) == 0:
        return False
    return True


def division_datasets(arquivos, classe):
    train_files, temp = sk.model_selection.train_test_split(
        arquivos, test_size=0.3, random_state=42
    )

    val_files, test_files = sk.model_selection.train_test_split(
        temp, test_size=0.5, random_state=42
    )

    for arquivo in train_files:
        shutil.copy2(
            os.path.join(dataset_dir, classe, arquivo),
            os.path.join(dir_train, classe)
        )

    for arquivo in val_files:
        shutil.copy2(
            os.path.join(dataset_dir, classe, arquivo),
            os.path.join(dir_validation, classe)
        )

    for arquivo in test_files:
        shutil.copy2(
            os.path.join(dataset_dir, classe, arquivo),
            os.path.join(dir_test, classe)
        )


def estratificacao(arquivos, classe):
    if not verification(classe):

        print(f"\nRecriando dataset para: {classe}")

        delete_files(os.path.join(dir_train, classe))
        delete_files(os.path.join(dir_validation, classe))
        delete_files(os.path.join(dir_test, classe))

        division_datasets(arquivos, classe)

        print(" Concluído:")
        print(f"Train: {count_wavs(os.path.join(dir_train, classe))}")
        print(
            f"Validation: {count_wavs(os.path.join(dir_validation, classe))}")
        print(f"Test: {count_wavs(os.path.join(dir_test, classe))}")

    else:
        print(f"\n Estratificação já realizada para {classe}")
        print(f"Train: {count_wavs(os.path.join(dir_train, classe))}")
        print(
            f"Validation: {count_wavs(os.path.join(dir_validation, classe))}")
        print(f"Test: {count_wavs(os.path.join(dir_test, classe))}")


dados = {
    'kick': list_wavs(os.path.join(dataset_dir, 'kick')),
    'snare': list_wavs(os.path.join(dataset_dir, 'snare')),
    'toms': list_wavs(os.path.join(dataset_dir, 'toms'))
}

for classe, arquivos in dados.items():
    estratificacao(arquivos, classe)
