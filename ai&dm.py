# Step 1: Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import pandas as pd

# Step 2: Load Iris dataset
iris = load_iris()
X_iris = iris.data  # Features
y_iris = iris.target  # Labels (species)

# Step 3: Load Wholesale Customers dataset
# Wholesale Customers dataset URL: https://archive.ics.uci.edu/ml/datasets/Wholesale+customers
# You may download it locally or use pandas to load it from a file (assuming it's named 'wholesale_customers.csv')
wholesale_df = pd.read_csv('C:\\Users\\aayus\\Downloads\\wholesale+customers\\Wholesale customers data.csv')
X_wholesale = wholesale_df.iloc[:, :-1].values  # Features (columns except for the last one)
y_wholesale = wholesale_df.iloc[:, -1].values  # Labels (Region)

# Step 4: Standardize the datasets
scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)
X_wholesale_scaled = scaler.fit_transform(X_wholesale)

# Step 5: Apply DBSCAN on both datasets
dbscan_iris = DBSCAN(eps=0.6, min_samples=5)
dbscan_iris_labels = dbscan_iris.fit_predict(X_iris_scaled)

dbscan_wholesale = DBSCAN(eps=0.6, min_samples=5)
dbscan_wholesale_labels = dbscan_wholesale.fit_predict(X_wholesale_scaled)

# Step 6: Apply k-NN on both datasets
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris_scaled, y_iris, test_size=0.3, random_state=42)
knn_iris = KNeighborsClassifier(n_neighbors=5)
knn_iris.fit(X_train_iris, y_train_iris)
knn_iris_pred = knn_iris.predict(X_test_iris)
knn_iris_accuracy = accuracy_score(y_test_iris, knn_iris_pred)
print(f'k-NN Accuracy on Iris dataset: {knn_iris_accuracy}')

X_train_wholesale, X_test_wholesale, y_train_wholesale, y_test_wholesale = train_test_split(X_wholesale_scaled, y_wholesale, test_size=0.3, random_state=42)
knn_wholesale = KNeighborsClassifier(n_neighbors=5)
knn_wholesale.fit(X_train_wholesale, y_train_wholesale)
knn_wholesale_pred = knn_wholesale.predict(X_test_wholesale)
knn_wholesale_accuracy = accuracy_score(y_test_wholesale, knn_wholesale_pred)
print(f'k-NN Accuracy on Wholesale dataset: {knn_wholesale_accuracy}')

# Step 7: Reduce dimensions using PCA for visualization (for both datasets)
pca_iris = PCA(n_components=2)
X_iris_pca = pca_iris.fit_transform(X_iris_scaled)

pca_wholesale = PCA(n_components=2)
X_wholesale_pca = pca_wholesale.fit_transform(X_wholesale_scaled)

# Step 8: Plot DBSCAN and k-NN results for both datasets
plt.figure(figsize=(14, 8))

# DBSCAN on Iris dataset
plt.subplot(2, 2, 1)
unique_labels_iris = set(dbscan_iris_labels)
colors = ['blue', 'green', 'orange', 'red']
for label in unique_labels_iris:
    if label == -1:  # Noise
        color = 'red'
        marker = 'x'
        label_name = 'Noise'
    else:
        color = colors[label % len(colors)]
        marker = 'o'
        label_name = f'Cluster {label}'
    
    plt.scatter(X_iris_pca[dbscan_iris_labels == label, 0], X_iris_pca[dbscan_iris_labels == label, 1], 
                c=color, label=label_name, marker=marker)

plt.title('DBSCAN Clustering on Iris Dataset')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.legend()

# k-NN on Iris dataset
plt.subplot(2, 2, 2)
for i, color in zip([0, 1, 2], ['blue', 'green', 'orange']):
    plt.scatter(X_iris_pca[y_iris == i, 0], X_iris_pca[y_iris == i, 1], c=color, label=iris.target_names[i], marker='o')

plt.title('k-NN Classification on Iris Dataset')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.legend()

# DBSCAN on Wholesale Customers dataset
plt.subplot(2, 2, 3)
unique_labels_wholesale = set(dbscan_wholesale_labels)
for label in unique_labels_wholesale:
    if label == -1:  # Noise
        color = 'red'
        marker = 'x'
        label_name = 'Noise'
    else:
        color = colors[label % len(colors)]
        marker = 'o'
        label_name = f'Cluster {label}'
    
    plt.scatter(X_wholesale_pca[dbscan_wholesale_labels == label, 0], X_wholesale_pca[dbscan_wholesale_labels == label, 1], 
                c=color, label=label_name, marker=marker)

plt.title('DBSCAN Clustering on Wholesale Dataset')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.legend()

# k-NN on Wholesale Customers dataset
plt.subplot(2, 2, 4)
plt.scatter(X_wholesale_pca[:, 0], X_wholesale_pca[:, 1], c=y_wholesale, cmap='viridis', marker='o')
plt.title('k-NN Classification on Wholesale Dataset')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')

plt.tight_layout()
plt.show()
