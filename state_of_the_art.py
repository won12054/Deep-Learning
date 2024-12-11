import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow.keras.applications import EfficientNetV2S 
import math
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from collections import Counter
import random

data_dir = 'C:/Users/Ale/CENTENNIAL/FALL_2024/Deep_learning/scripts/train'
image_size = (260, 260)  
batch_size = 32

images = []
labels = []
class_labels = os.listdir(data_dir)  
class_mapping = {class_labels[i]: i for i in range(len(class_labels))}

for label in class_labels:
    class_dir = os.path.join(data_dir, label)
    for file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, file)
        img = load_img(img_path, target_size=image_size)
        img_array = img_to_array(img)
        images.append(img_array)
        labels.append(class_mapping[label])

images = np.array(images) / 255.0  
labels = np.array(labels)

X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, stratify=labels, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

y_train = to_categorical(y_train, num_classes=2)
y_val = to_categorical(y_val, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

print(f"Training set: {len(y_train)}")
print(f"Validation set: {len(y_val)}")
print(f"Test set: {len(y_test)}")

train_datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
)

val_test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(
    X_train, y_train, batch_size=batch_size
)

validation_generator = val_test_datagen.flow(
    X_val, y_val, batch_size=batch_size
)

test_generator = val_test_datagen.flow(
    X_test, y_test, batch_size=batch_size, shuffle=False
)

def generator_to_tfdata(generator):
    def gen():
        for x, y in generator:
            yield x, y
    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, 260, 260, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32)
        )
    )

train_dataset = generator_to_tfdata(train_generator).repeat() 
validation_dataset = generator_to_tfdata(validation_generator).repeat()

base_model = EfficientNetV2S(weights='imagenet', include_top=False, input_shape=(260, 260, 3))
base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation='relu')(x)
predictions = layers.Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1,
    restore_best_weights=True)

early_stopping_fine = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True)

model.summary()

history = model.fit(
    train_dataset,
    steps_per_epoch=math.ceil(len(X_train) / batch_size),  
    epochs=5,
    validation_data=validation_dataset,
    validation_steps=math.ceil(len(X_val) / batch_size),  
    callbacks=[early_stopping]
)

def plot_history(history, title_suffix=''):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Model accuracy {title_suffix}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model loss {title_suffix}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig('results/sota_training_history.png')
    plt.show()

plot_history(history, title_suffix='(Initial Training)')

base_model.trainable = True
fine_tune_at = int(len(base_model.layers) * 0.8)
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history_fine = model.fit(
    train_dataset,
    steps_per_epoch=math.ceil(len(X_train) / batch_size),  
    epochs=30,
    validation_data=validation_dataset,
    validation_steps=math.ceil(len(X_val) / batch_size),  
    callbacks=[early_stopping_fine])

plot_history(history_fine, title_suffix='(Fine-Tuning)')

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

predictions = model.predict(test_generator, steps=len(test_generator))
predicted_classes = np.argmax(predictions, axis=1)

true_classes = np.argmax(y_test, axis=1)   

class_labels = list(test_generator.class_indices.keys()) if hasattr(test_generator, 'class_indices') else ['Class 0', 'Class 1']

report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

cm = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:")
print(cm)

roc_auc = roc_auc_score(true_classes, predictions[:, 1])
print("ROC-AUC Score:", roc_auc)

fpr, tpr, thresholds = roc_curve(true_classes, predictions[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('results/sota_roc.png')
plt.show()

class_labels = ['Healthy', 'Sick']
label_map = {i: label for i, label in enumerate(class_labels)}  

random_indices = random.sample(range(len(X_test)), min(9, len(X_test)))  

all_images = X_test
all_labels = y_test

plt.figure(figsize=(10, 10))
for i, idx in enumerate(random_indices):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(all_images[idx])

    predicted_label = label_map[np.argmax(predictions[idx])]  
    actual_label = label_map[np.argmax(all_labels[idx])]      
    prediction_confidence = np.max(predictions[idx]) * 100   

    plt.title(f"Pred: {predicted_label} ({prediction_confidence:.2f}%)\nActual: {actual_label}")
    plt.axis("off")
plt.suptitle("Images with Predictions and Confidence", fontsize=16)
plt.tight_layout()
plt.savefig('results/sota_predictions.png')
plt.show()

