import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import librosa as lb
import os
import sklearn as sk
import seaborn as sns
import pandas as pd


batch_size = 32
n_mels = 128
n_fft = 2048
hop_length = 512
win_length = 2048
fmax = None
sample_rate = 44100

train = []
test = []
validation = []

classes = {"kick": 0, "snare": 1, "toms": 2}


def duration_care(y):

    duration_difference = (sample_rate * 2) - y.shape[0]
    if (duration_difference > 0):
        print("Padding do audio com zeros para 2 seconds.")
        y = np.pad(y, (0, duration_difference), mode='constant')
    if (duration_difference < 0):
        print("Cortando o audio para 2 seconds.")
        y = y[:sample_rate*2]

    return y

def data_augmentation(name_step):
    data_augmented = []
    vector = []

    for classe, valor in classes.items():
        path = "./Dataset/" + name_step + "/" + classe + "/"
        for audio in sorted(os.listdir(path)):
            if audio.endswith(".wav"):
                # carregamento do áudio
                y, sr = lb.load(os.path.join(path, audio), sr=sample_rate)

                if (cenary != "1"):
                    if (cenary == "3"):
                        steps = np.random.choice(
                            [-1.0, -0.5, 0.5, 1.0], replace=False, size=2)
                        for step in steps:
                            # modificação do áudio para o data augmentation
                            y_aug = lb.effects.pitch_shift(
                                y, sr=sample_rate, n_steps=float(step))

                            # Tratamento para normalização da duração do áudio em 2 segundos (44100*2 amostras)
                            y_aug = duration_care(y_aug)

                            # Tupla com os dados de áudio modificados e o valor da classe correspondente
                            data_augmented.append((y_aug, valor))
                    else:      
                        step = np.random.choice(
                            [-1.0, -0.5, 0.5, 1.0])
                        # modificação do áudio para o data augmentation
                        y_aug = lb.effects.pitch_shift(
                            y, sr=sample_rate, n_steps=float(step))

                        # Tratamento para normalização da duração do áudio em 2 segundos (44100*2 amostras)
                        y_aug = duration_care(y_aug)

                        # Tupla com os dados de áudio modificados e o valor da classe correspondente
                        data_augmented.append((y_aug, valor))

    for audio_augment, rotulo in data_augmented:
        melspec = lb.feature.melspectrogram(
            y=audio_augment,
            sr=sample_rate,
            n_mels=n_mels,
            fmax=fmax,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

        # conversão para escala logarítmica
        melspec_log = lb.power_to_db(melspec, ref=np.max)

        # retorno da tupla (espectrograma de Mel, valor da classe)
        vector.append((melspec_log, rotulo))

    return vector


def melspectrogram(name_step, vector):

    for classe, valor in classes.items():
        path = "./Dataset/" + name_step + "/" + classe + "/"
        for audio in sorted(os.listdir(path)):
            if audio.endswith(".wav"):

               # carregamento do áudio
                y, sr = lb.load(os.path.join(path, audio), sr=sample_rate)

                # tratamento para a duração do áudio ser padronizada em 2 segundos (44100*2 amostras)
                y = duration_care(y)

                # cálculo do espectrograma de Mel
                melspec = lb.feature.melspectrogram(
                    y=y,
                    sr=sample_rate,
                    n_mels=n_mels,
                    fmax=fmax,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                )

                # conversão para escala logarítmica
                melspec_log = lb.power_to_db(melspec, ref=np.max)

                # retorno da tupla (espectrograma de Mel, valor da classe)
                vector.append((melspec_log, valor))

    return vector


def prepare_to_dataset(melspc_vector):
    x = []
    y = []
    for i, j in melspc_vector:
        x.append(i)
        y.append(j)
    x = np.array(x, dtype=np.float32)
    x = x[..., np.newaxis]
    y = np.array(y, dtype=np.int32)
    return x, y


def menu():
    print("Escolha o cenário de data augmentation para ser executado 3 vezes:")
    print("1. Sem data augmentation (1x)")
    print("2. Data augmentation com 1 variação (2x)")
    print("3. Data augmentation com 2 variações (3x)")
    choice = input("Digite o número correspondente à sua escolha: ")
    return choice


cenary = menu()

cases_seeds = [42, 123, 2024]

for i in range(3):
    train = []
    test = []
    validation = []

    seed = cases_seeds[i]
    tf.random.set_seed(seed)
    np.random.seed(seed)

    train = melspectrogram("train", train)

    if (cenary != "1"):
        data_aug = data_augmentation("train")
        train.extend(data_aug)
        print(f"Quantidade de amostras com data pitch_shift: {len(data_aug)}")

    coluna = [item[0].tobytes() for item in train]
    if (coluna != []):
        print(
            f"Quantidade de amostras únicas com data augmentation: {len(set(coluna))}")
    print(f"Quantidade de amostras de treinamento: {len(train)}")
    test = melspectrogram("test", test)
    validation = melspectrogram("validation", validation)


    train_x, train_labels = prepare_to_dataset(train)
    validation_x, validation_labels = prepare_to_dataset(validation)
    test_x, test_labels = prepare_to_dataset(test)

    num_classes = len(classes)

    AUTOTUNE = tf.data.AUTOTUNE

    train = tf.data.Dataset.from_tensor_slices((train_x, train_labels))
    train = (
        train.shuffle(buffer_size=len(train_x), seed=seed)
        .batch(batch_size)
        .cache()
        .prefetch(AUTOTUNE)
    )
    validation = tf.data.Dataset.from_tensor_slices(
        (validation_x, validation_labels))
    validation = validation.batch(batch_size).cache().prefetch(AUTOTUNE)
    test = tf.data.Dataset.from_tensor_slices((test_x, test_labels))
    test = test.batch(batch_size).cache().prefetch(AUTOTUNE)

    print(train.element_spec)


    norm_layer = layers.Normalization()
    norm_layer.adapt(train_x)
    model = models.Sequential(
        [
            layers.Input(shape=(train_x.shape[1:])),
            norm_layer,
            layers.Conv2D(16, 3, activation="relu"),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes),
        ]
    )

    model.summary()

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )

    epochs = 10
    history = model.fit(
        train,
        validation_data=validation,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                verbose=1, patience=2, restore_best_weights=True
            )
        ],
    )

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    metrics = history.history
    
    if (cenary == "1"):
        type_cenary = "Sem_data_augmentation(1x)"
    elif (cenary == "2"):
        type_cenary = "Com_Data_augmentation(2x)"
    elif (cenary == "3"):
        type_cenary = "Com_Data_augmentation(3x)"

    plt.figure(figsize=(10, 5))
    plt.plot(history.epoch, metrics["loss"])
    plt.plot(history.epoch, metrics["val_loss"])
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.title(f"Loss de Treinamento e Validação {type_cenary} execução {i+1}")
    plt.legend(["Treinamento", "Validação"])
    plt.grid(True)
    plt.savefig(
        f"./Resultados/cenario{cenary}/training_Loss_{type_cenary}_execucao{i+1}.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history.epoch, metrics["accuracy"])
    plt.plot(history.epoch, metrics["val_accuracy"])
    plt.xlabel("Épocas")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy de Treinamento e Validação {type_cenary} execução {i+1}")
    plt.legend(["Treinamento", "Validação"])
    plt.grid(True)
    plt.savefig(
        f"./Resultados/cenario{cenary}/training_accuracy_{type_cenary}_execucao{i+1}.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()

    test_audio = []
    test_labels = []

    for matriz, labels in test:
        test_audio.append(matriz)
        test_labels.append(labels)
    test_audio = np.concatenate(test_audio, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_true = test_labels

    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f"Test Accuracy: {test_acc:.0%}")

    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)

    class_names = list(classes.keys())

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_mtx, xticklabels=class_names, yticklabels=class_names, annot=True, fmt="d"
    )
    plt.xlabel("Prediction")
    plt.ylabel("Label")
    plt.title(f"Confusion Matrix\n Test Accuracy: {test_acc:.0%}")
    plt.savefig(f"./Resultados/cenario{cenary}/confusion_matrix_{type_cenary}_execucao{i+1}.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    results = sk.metrics.classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True)

    df = pd.DataFrame(results).transpose()

    df.to_csv(
        f"./Resultados/cenario{cenary}/classification_report_{type_cenary}_execucao{i+1}.csv", index=True)
    print(df)



