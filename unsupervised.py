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

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
print(f"Train set: {len(train_images)}, Test set: {len(test_images)}")

print("Extracting features using ResNet50...")
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
base_model.summary()

train_features = feature_extractor.predict(train_images)
train_features_flattened = train_features.reshape(train_features.shape[0], -1)

test_features = feature_extractor.predict(test_images)
test_features_flattened = test_features.reshape(test_features.shape[0], -1)

print(f"Shape of train_features: {train_features.shape}")
print(f"Shape of train_features_flattened: {train_features_flattened.shape}")

tsne = TSNE(n_components=2, random_state=42)
reduced_features = tsne.fit_transform(train_features_flattened)

plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=train_labels, cmap='viridis', s=10)
plt.title("t-SNE Visualization of ResNet Features (Training)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.colorbar(label="True Labels")
plt.savefig('results/unsupervised_training_resnet.png')
plt.show()

'''
PCA needs to be applied! Memory Error due to many dimensions.
'''
pca = PCA(n_components=0.99, svd_solver='full')  
train_features_pca = pca.fit_transform(train_features_flattened)
print(f"Shape after PCA: {train_features_pca.shape}")

test_features_pca = pca.transform(test_features_flattened)  
print(f"Shape after PCA (Test Set): {test_features_pca.shape}")

tsne = TSNE(n_components=2, random_state=42)
tsne_features = tsne.fit_transform(train_features_pca)

plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=train_labels, cmap='viridis', s=10)
plt.title("t-SNE Visualization after PCA (Training)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.colorbar(label="True Labels")
plt.savefig('results/unsupervised_training_pca.png')
plt.show()

cluster_range = range(2, 10)
silhouette_scores = []

for num_clusters in cluster_range:
    gmm = GaussianMixture(n_components=num_clusters, random_state=42)
    gmm_clusters = gmm.fit_predict(train_features_pca)
    score = silhouette_score(train_features_pca, gmm_clusters)
    silhouette_scores.append(score)

plt.figure(figsize=(10, 6))
plt.plot(cluster_range, silhouette_scores, marker='o')
plt.title("Silhouette Score vs. Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.grid()
plt.savefig('results/unsupervised_silhouette_vs_num_cluster.png')
plt.show()

'''
As we have 2 labels, I will set the number of clusters to two
'''
num_clusters = 2
gmm = GaussianMixture(n_components=num_clusters, random_state=42)
gmm_clusters = gmm.fit_predict(train_features_pca)

sil_score = silhouette_score(train_features_pca, gmm_clusters)
print(f"Silhouette Score: {sil_score:.4f}")

print("Analyzing cluster-label relationship...")
for cluster_id in range(num_clusters):
    cluster_indices = np.where(gmm_clusters == cluster_id)[0]  
    cluster_labels = train_labels[cluster_indices] 
    print(f"Cluster {cluster_id}:")
    print(f"  Total samples: {len(cluster_indices)}")
    print(f"  Healthy: {np.sum(cluster_labels == 0)}")
    print(f"  Sick: {np.sum(cluster_labels == 1)}")

plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=gmm_clusters, cmap='viridis', s=10)
plt.title("t-SNE Visualization of GMM Clusters (Training)")
plt.colorbar(label="Cluster")
plt.savefig('results/unsupervised_training_gmm.png')
plt.show()

print("Applying GMM to Test Set...")
test_clusters = gmm.predict(test_features_pca)  

print("Analyzing Test Set cluster-label relationship...")
for cluster_id in range(num_clusters):
    cluster_indices = np.where(test_clusters == cluster_id)[0]  
    cluster_labels = test_labels[cluster_indices]  
    print(f"Cluster {cluster_id}:")
    print(f"  Total samples: {len(cluster_indices)}")
    print(f"  Healthy: {np.sum(cluster_labels == 0)}")
    print(f"  Sick: {np.sum(cluster_labels == 1)}")

print("Training supervised model using extracted features...")
clf = RandomForestClassifier(random_state=42)
clf.fit(train_features_pca, train_labels)

print("Evaluating supervised model on Test Set...")
y_pred = clf.predict(test_features_pca)  
y_pred_proba = clf.predict_proba(test_features_pca)[:, 1]  

print("Classification Report:")
print(classification_report(test_labels, y_pred))

roc_auc = roc_auc_score(test_labels, y_pred_proba)
print(f"ROC AUC Score: {roc_auc:.4f}")

fpr, tpr, _ = roc_curve(test_labels, y_pred_proba)

plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid()
plt.savefig('results/unsupervised_roc.png')
plt.show()

print("Confusion Matrix:")
cm = confusion_matrix(test_labels, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', 'Sick'])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig('results/unsupervised_confusion_matrix.png')
plt.show()
