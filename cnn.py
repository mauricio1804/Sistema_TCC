import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import librosa as lb
import os
import sklearn as sk

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

batch_size = 32

train = []
test = []
validation = []

classes = ['kick', 'snare', 'toms']

def melspectrogram(name_step, vector):
    
    for classe in classes:
        path = './Dataset/' + name_step + '/' + classe + '/'
        for audio in os.listdir(path):
            if audio.endswith('.wav'):
                y, sr = lb.load(os.path.join(path, audio), sr=None)
                melspec = lb.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=None, n_fft=2048, hop_length=512)
                melspec_log = lb.power_to_db(melspec, ref=np.max)
                vector.append((melspec_log, classe))
                
    return vector

train = melspectrogram('train', train)
test = melspectrogram('test', test)
validation = melspectrogram('validation', validation)

def prepare_to_dataset(melspc_vector):
    x = []
    y = []
    for i, j in melspc_vector:
        x.append(i)
        y.append(j)
    x = np.array(x)
    x = x[..., np.newaxis]
    y = np.array(y)
    y = sk.preprocessing.LabelEncoder().fit_transform(y)
    y = tf.keras.utils.to_categorical(y)
    return x, y

train_x, train_labels = prepare_to_dataset(train)
validation_x, validation_labels = prepare_to_dataset(validation)
test_x, test_labels = prepare_to_dataset(test)

num_classes = train_labels.shape[1]


AUTOTUNE = tf.data.AUTOTUNE

train = tf.data.Dataset.from_tensor_slices((train_x, train_labels))
train = train.shuffle(buffer_size=len(train_x), seed=seed).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
validation = tf.data.Dataset.from_tensor_slices((validation_x, validation_labels))
validation = validation.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
test = tf.data.Dataset.from_tensor_slices((test_x, test_labels))
test = test.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

print(train.element_spec)


norm_layer = layers.Normalization()
norm_layer.adapt(train_x)
model = models.Sequential([
    layers.Input(shape=(train_x.shape[1:])),
    layers.Resizing(32, 32),
    norm_layer,
    layers.Conv2D(16, 3, activation='relu'),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision", top_k=1),
        tf.keras.metrics.Recall(name="recall", top_k=1),
        tf.keras.metrics.F1Score(name="f1_macro", average="macro", threshold=None),
    ],
)

epochs = 10
history = model.fit(
    train,
    validation_data=validation,
    epochs=epochs,
    callbacks=[tf.keras.callbacks.EarlyStopping(verbose=1, patience=2)]
)

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

metrics = history.history
plt.figure(figsize=(10, 5))
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(history.epoch, metrics['accuracy'], metrics['val_accuracy'])
plt.legend(['accuracy', 'val_accuracy'])
plt.savefig('training_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(history.epoch, metrics['recall'], metrics['val_recall'])
plt.legend(['recall', 'val_recall'])
plt.savefig('training_recall.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(history.epoch, metrics['f1_macro'], metrics['val_f1_macro'])
plt.legend(['f1_macro', 'val_f1_macro'])
plt.savefig('training_f1_score.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(history.epoch, metrics['precision'], metrics['val_precision'])
plt.legend(['precision', 'val_precision'])
plt.savefig('training_precision.png', dpi=150, bbox_inches='tight')
plt.close()

y_true = []
y_pred = []

for matriz, labels in test:
    logits = model(matriz, training=False)
    predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
    true_labels = tf.argmax(labels, axis=1, output_type=tf.int32)

    y_true.append(true_labels)
    y_pred.append(predictions)

y_true = tf.concat(y_true, axis=0)
y_pred = tf.concat(y_pred, axis=0)

cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes)

confusion_mtx = cm.numpy()

plt.figure(figsize=(10, 8))
plt.imshow(confusion_mtx, cmap='Blues')
plt.colorbar()
plt.xticks(ticks=range(num_classes), labels=classes)
plt.yticks(ticks=range(num_classes), labels=classes)
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, str(confusion_mtx[i, j]), ha='center', va='center', color='black')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()


results = model.evaluate(test, return_dict=True)


