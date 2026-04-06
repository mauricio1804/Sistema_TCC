import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os

batch_size = 32
img_height = 480
img_width = 640

train = tf.keras.utils.image_dataset_from_directory(
    './Dataset/train/',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical'
)

validation = tf.keras.utils.image_dataset_from_directory(
    './Dataset/validation/',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical'
)
test = tf.keras.utils.image_dataset_from_directory(
    './Dataset/test/',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical'
)

class_names = train.class_names
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train = train.cache().shuffle(84).prefetch(buffer_size=AUTOTUNE)
validation = validation.cache().prefetch(buffer_size=AUTOTUNE)
test = test.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

norm_layer = layers.Normalization()
norm_layer.adapt(data=train.map(lambda spec, label: spec))

model = models.Sequential([
    layers.Input(shape=(img_height, img_width, 3)),
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

for images, labels in test:
    logits = model(images, training=False)
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
plt.xticks(ticks=range(num_classes), labels=class_names)
plt.yticks(ticks=range(num_classes), labels=class_names)
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, str(confusion_mtx[i, j]), ha='center', va='center', color='black')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()


results = model.evaluate(test, return_dict=True)
print(results)

