# -*- coding: utf-8 -*-


import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, adjusted_rand_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf

import itertools
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

DATASET_DIR = 'C:/Users/Ale/CENTENNIAL/FALL_2024/Deep_learning/scripts/train'
def load_images_and_labels(dataset_dir):
    images = []
    labels = []
    categories = ['Healthy', 'Sick']

    for label, category in enumerate(categories):
        category_path = os.path.join(dataset_dir, category)
        for img_name in os.listdir(category_path):
            if img_name.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(category_path, img_name)
                img = load_img(img_path, target_size=(224, 224))
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(label)
    return np.array(images), np.array(labels)

print("Loading dataset...")
images, labels = load_images_and_labels(DATASET_DIR)
print(f"Loaded {len(images)} images.")

images.max()
images.min()

X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
print(f"Train set: {len(X_train)}, Val set: {len(X_val)}, Test set: {len(X_test)}")

# best hyperparameters is 
# layers option = [64,128,256]
# dropout = 0.3
# learning rate = 0.001
layers_options =[[64,128,256]]
dropout_options =[0.3]
learning_rate_options = [0.001]

results = []

def create_cnn_model(X_train, y_train):
    for layers, dropout_rate, lr in itertools.product(layers_options, dropout_options, learning_rate_options):
        print(f'Testing for: Layers={layers}, Dropout={dropout_rate}, Learning rate={lr}')
        model = Sequential()
        for filters in layers:
            model.add(Conv2D(filters,(3,3), activation='relu',padding='same'))
            model.add(MaxPooling2D(2,2))
        model.add(Flatten())
        model.add(Dense(128,activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1,activation='sigmoid'))
        #compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
        #callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3)
        #train the model
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[early_stopping, reduce_lr], verbose=1)
        train_accuracy = history.history['accuracy'][-1]
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        val_accuracy = history.history['val_accuracy'][-1]
        #record results
        results.append({
        'model':model,
        'layers':layers,
        'dropout':dropout_rate,
        'learning_rate':lr,
        'history':history,
        'val_loss':val_loss,
        'val_accuracy':val_accuracy})
        print(f'Train Loss: {train_loss}')
        print(f'Train Accuracy: {train_accuracy}')
        print(f'Validation Loss: {val_loss}')
        print(f'Validation Accuracy: {val_accuracy}')
    return results

model_results = create_cnn_model(X_train, y_train)

print(model_results)
best_model_config = max(model_results, key=lambda x:x['val_accuracy'])
print(best_model_config)
best_model = best_model_config['model']
best_model.summary()
best_history = best_model_config['history']

print(f'Training Accuracy: {best_model_config["history"].history["accuracy"][-1]}')
print(f'Validation Accuracy: {best_model_config["history"].history["val_accuracy"][-1]}')

import matplotlib.pyplot as plt
train_accuracy = best_history.history['accuracy']
val_accuracy = best_history.history['val_accuracy']
train_loss = best_history.history['loss']
val_loss = best_history.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.savefig('results/supervised_training_history.png')
plt.show()



test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

predictions = best_model.predict(X_test)
binary_predictions = (predictions > 0.5).astype(int)

print('Classification Report')
print(classification_report(y_test, binary_predictions))

print('Confusion Matrix')
cm = confusion_matrix(y_test, binary_predictions)
print(cm)

fpr, tpr, thresholds = roc_curve(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)
print(f'ROC AUC: {roc_auc:.2f}')

plt.figure(figsize=(8,6))
plt.plot(fpr,tpr, color='blue', label='ROC Curve')
plt.plot([0,1],[0,1], color='red', linestyle='--')
plt.title('ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig('results/supervised_roc.png')
plt.show()

class_labels = ['Healthy', 'Sick']
label_map = {i: label for i, label in enumerate(class_labels)}
binary_predictions = binary_predictions.flatten()


random_indices = np.random.choice(len(X_test), size=9, replace=False)
all_images = X_test
all_labels = y_test
plt.figure(figsize=(10,10))


for i, idx in enumerate(random_indices):
    ax = plt.subplot(3,3,i+1)
    plt.imshow(all_images[idx])
    predicted_label = label_map[binary_predictions[idx]]
    if len(all_labels.shape) == 1:
        actual_label = label_map[all_labels[idx]]
    else:
        actual_label = label_map(np.argmax(all_labels[idx]))
    if predictions[idx].ndim ==0:
        prediction_confidence = predictions[idx] * 100
    else:
        prediction_confidence = np.max(predictions[idx])*100

    plt.title(f"Actual: {actual_label}\nPredicted: {predicted_label}\nConfidence: {prediction_confidence:.2f}%",fontsize=10)
    plt.axis('off')
plt.suptitle("Images with Predictions and Confidence", fontsize=10)
plt.tight_layout()
plt.savefig('results/supervised_predictions.png')
plt.show()

