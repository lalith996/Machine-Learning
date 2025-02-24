import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
import matplotlib.pyplot as plt

# Load dataset
def load_data(file_path):
    """
    Load dataset from an Excel file.
    Args:
        file_path: Path to the Excel file.
    Returns:
        df: Loaded DataFrame.
    """
    df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")
    return df

# Preprocess data
def preprocess_data(df):
    """
    Preprocess the dataset by handling missing values and encoding categorical variables.
    Args:
        df: Input DataFrame.
    Returns:
        X: Feature matrix.
        y: Target variable.
    """
    # Drop rows with missing target values
    df = df.dropna(subset=["Condition"])
    
    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)
    
    # Separate features and target
    X = df.drop(columns=["Condition_NO CONDITION", "Condition_S", "Condition_AK", "Condition_R", "Condition_I", "Condition_M", "Condition_F", "Condition_N"])
    y = df["Condition_NO CONDITION"]  # Assuming binary classification for simplicity
    return X, y

# Train-test split
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    Args:
        X: Feature matrix.
        y: Target variable.
        test_size: Proportion of test data.
        random_state: Random seed for reproducibility.
    Returns:
        X_train, X_test, y_train, y_test: Split datasets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Evaluate regression metrics
def evaluate_regression_metrics(y_true, y_pred):
    """
    Calculate regression metrics: MSE, RMSE, MAPE, R2.
    Args:
        y_true: True target values.
        y_pred: Predicted target values.
    Returns:
        metrics: Dictionary containing MSE, RMSE, MAPE, R2 scores.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2}

# Train Linear Regression model
def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.
    Args:
        X_train: Training feature matrix.
        y_train: Training target variable.
    Returns:
        reg: Trained Linear Regression model.
    """
    reg = LinearRegression().fit(X_train, y_train)
    return reg

# Perform k-means clustering
def perform_kmeans(X_train, n_clusters=2):
    """
    Perform k-means clustering.
    Args:
        X_train: Feature matrix.
        n_clusters: Number of clusters.
    Returns:
        kmeans: Trained KMeans model.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit(X_train)
    return kmeans

# Evaluate clustering metrics
def evaluate_clustering_metrics(X_train, labels):
    """
    Evaluate clustering metrics: Silhouette Score, CH Score, DB Index.
    Args:
        X_train: Feature matrix.
        labels: Cluster labels.
    Returns:
        metrics: Dictionary containing Silhouette Score, CH Score, DB Index.
    """
    silhouette = silhouette_score(X_train, labels)
    ch_score = calinski_harabasz_score(X_train, labels)
    db_index = davies_bouldin_score(X_train, labels)
    return {"Silhouette Score": silhouette, "CH Score": ch_score, "DB Index": db_index}

# Main program
if __name__ == "__main__":
    # Load and preprocess data
    file_path = "/Users/lalithmachavarapu/Downloads/Lab Session Data.xlsx"
    df = load_data(file_path)
    X, y = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Task A1: Train Linear Regression model with one feature
    reg_one_feature = train_linear_regression(X_train[["age"]], y_train)
    y_train_pred_one = reg_one_feature.predict(X_train[["age"]])
    y_test_pred_one = reg_one_feature.predict(X_test[["age"]])
    
    # Task A2: Evaluate regression metrics for one feature
    train_metrics_one = evaluate_regression_metrics(y_train, y_train_pred_one)
    test_metrics_one = evaluate_regression_metrics(y_test, y_test_pred_one)
    print("Metrics for one feature (Train):", train_metrics_one)
    print("Metrics for one feature (Test):", test_metrics_one)
    
    # Task A3: Train Linear Regression model with all features
    reg_all_features = train_linear_regression(X_train, y_train)
    y_train_pred_all = reg_all_features.predict(X_train)
    y_test_pred_all = reg_all_features.predict(X_test)
    
    # Evaluate regression metrics for all features
    train_metrics_all = evaluate_regression_metrics(y_train, y_train_pred_all)
    test_metrics_all = evaluate_regression_metrics(y_test, y_test_pred_all)
    print("Metrics for all features (Train):", train_metrics_all)
    print("Metrics for all features (Test):", test_metrics_all)
    
    # Task A4: Perform k-means clustering
    kmeans = perform_kmeans(X_train, n_clusters=2)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    print("Cluster Labels:", cluster_labels)
    print("Cluster Centers:", cluster_centers)
    
    # Task A5: Evaluate clustering metrics
    clustering_metrics = evaluate_clustering_metrics(X_train, cluster_labels)
    print("Clustering Metrics:", clustering_metrics)
    
    # Task A6: Perform k-means for different k values
    k_values = range(2, 11)
    silhouette_scores = []
    ch_scores = []
    db_indices = []
    
    for k in k_values:
        kmeans = perform_kmeans(X_train, n_clusters=k)
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(X_train, labels))
        ch_scores.append(calinski_harabasz_score(X_train, labels))
        db_indices.append(davies_bouldin_score(X_train, labels))
    
    # Plot clustering metrics
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.title("Silhouette Score vs k")
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    
    plt.subplot(1, 3, 2)
    plt.plot(k_values, ch_scores, marker='o')
    plt.title("CH Score vs k")
    plt.xlabel("k")
    plt.ylabel("CH Score")
    
    plt.subplot(1, 3, 3)
    plt.plot(k_values, db_indices, marker='o')
    plt.title("DB Index vs k")
    plt.xlabel("k")
    plt.ylabel("DB Index")
    
    plt.tight_layout()
    plt.show()
    
    # Task A7: Elbow method for optimal k
    distortions = []
    for k in range(2, 21):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X_train)
        distortions.append(kmeans.inertia_)
    
    plt.figure(figsize=(6, 4))
    plt.plot(range(2, 21), distortions, marker='o')
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("k")
    plt.ylabel("Distortion")
    plt.show()