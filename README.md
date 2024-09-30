# DBSCAN vs k-NN: Comparative Analysis using Real-World Datasets

## Introduction
In this project, we compare two machine learning algorithms: **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) and **k-NN** (k-Nearest Neighbors) using two real-world datasets: **Iris** and **Wholesale Customers**. The goal is to understand the differences between a **density-based clustering algorithm** (DBSCAN) and a **distance-based classification algorithm** (k-NN) by visualizing how these algorithms work on different datasets.

## Datasets Used

### 1. Iris Dataset
- **Overview**: A dataset containing 150 samples of iris flowers, each represented by four features: sepal length, sepal width, petal length, and petal width.
- **Classes**: 3 species of iris flowers: Setosa, Versicolor, and Virginica.
- **Purpose**: To evaluate both clustering and classification on a simple, structured dataset where classes are relatively well-separated.

### 2. Wholesale Customers Dataset
- **Overview**: This dataset includes annual spending in monetary units by customers across six product categories: Fresh, Milk, Grocery, Frozen, Detergents_Paper, and Delicatessen.
- **Classes**: The `Region` column is used as the label for supervised classification.
- **Purpose**: Provides a more complex, real-world scenario, where classes are not easily separable.

## Algorithms

### 1. DBSCAN (Unsupervised Learning)
- **How it works**: DBSCAN groups data points that are close to each other in high-density regions. It identifies noise points that don't belong to any cluster.
- **Parameters**:
  - `eps`: Maximum distance between two points for them to be considered neighbors.
  - `min_samples`: Minimum number of points required to form a dense region (a cluster).
- **Strengths**: Finds clusters of varying shapes, handles noise, and doesn’t need to know the number of clusters beforehand.
- **Limitations**: Sensitive to parameter selection (`eps` and `min_samples`) and struggles with datasets with varying densities.

### 2. k-Nearest Neighbors (k-NN) (Supervised Learning)
- **How it works**: k-NN classifies data points based on the majority class of the `k` nearest neighbors.
- **Parameter**:
  - `k`: Number of neighbors considered for classification.
- **Strengths**: Simple, effective on small datasets with clear class boundaries.
- **Limitations**: Sensitive to noisy data, and can struggle with imbalanced datasets or overlapping classes.

## Supervised Algorithms That Could Be Used
Other supervised learning algorithms that could have been used include:
- **Decision Trees**: For splitting data into subsets based on feature values.
- **Support Vector Machines (SVM)**: To find a hyperplane that separates data classes.
- **Random Forest**: Builds multiple decision trees and aggregates their results.
- **Logistic Regression**: Predicts the probability of a class by fitting data to a logistic function.

**Why k-NN?**  
k-NN was chosen for this project because of its simplicity and flexibility in dealing with structured (Iris) and complex (Wholesale Customers) data. Its proximity-based classification is easy to understand and provides a clear contrast to DBSCAN’s density-based clustering.

## Simulation
![image](https://github.com/user-attachments/assets/6e21899d-2d27-4bae-a828-d1176bcf930f)

![image](https://github.com/user-attachments/assets/7ed9226e-d153-49d8-a163-f8f30c95c094)
