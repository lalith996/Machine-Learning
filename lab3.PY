import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Function to load data
def load_data(file_path, sheet_name):
    """Load data from the specified sheet in the Excel file."""
    try:
        data = pd.read_excel(file_path, sheet_name=sheet_name)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


# Function to preprocess data
def preprocess_data(data, label_column):
    """Preprocess the data for classification."""
    # Drop rows with missing values in the label column
    data = data.dropna(subset=[label_column])
    
    # Encode the label column (binary encoding)
    data['Label'] = data[label_column].apply(lambda x: 0 if x == "NO CONDITION" else 1)
    
    # Identify categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    # Apply Label Encoding to categorical columns
    label_encoders = {}
    for col in categorical_cols:
        if col != label_column:  # Skip the label column
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))  # Convert to string to handle NaNs
            label_encoders[col] = le
    
    # Select features and labels
    X = data.drop(columns=[label_column, 'Label']).values  # Exclude non-feature columns
    y = data['Label'].values
    
    return X, y


# Function to evaluate confusion matrix and metrics
def evaluate_confusion_matrix(y_true, y_pred):
    """Evaluate confusion matrix and performance metrics."""
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return cm, precision, recall, f1, accuracy


# Function to plot decision boundary
def plot_decision_boundary(X_train, y_train, X_test, y_test, k):
    """Plot decision boundary for kNN classifier."""
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    
    # Adjust step size to avoid excessive memory usage
    step_size = 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step_size),
        np.arange(y_min, y_max, step_size)
    )
    
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    colors = ['blue' if label == 0 else 'red' for label in y_train]
    plt.scatter(X_train[:, 0], X_train[:, 1], c=colors, edgecolor='k', s=100)
    plt.title(f"kNN Decision Boundary (k={k})")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.show()


# Function to tune hyperparameters
def tune_knn_hyperparameters(X_train, y_train):
    """Perform hyperparameter tuning using RandomizedSearchCV."""
    param_grid = {'n_neighbors': range(1, 21)}
    knn = KNeighborsClassifier()
    random_search = RandomizedSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_iter=10, random_state=42)
    random_search.fit(X_train, y_train)
    best_k = random_search.best_params_['n_neighbors']
    return best_k


# Main Program
if __name__ == "__main__":
    # Load data from the thyroid sheet
    file_path = "/Users/lalithmachavarapu/Downloads/Lab Session Data.xlsx"
    sheet_name = "thyroid0387_UCI"
    label_column = "Condition"
    
    data = load_data(file_path, sheet_name)
    if data is not None:
        X, y = preprocess_data(data, label_column)
        
        # A1: Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # A2: Train kNN classifier (k=3)
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        
        # Evaluate training and test performance
        y_train_pred = knn.predict(X_train)
        y_test_pred = knn.predict(X_test)
        
        train_cm, train_precision, train_recall, train_f1, train_accuracy = evaluate_confusion_matrix(y_train, y_train_pred)
        test_cm, test_precision, test_recall, test_f1, test_accuracy = evaluate_confusion_matrix(y_test, y_test_pred)
        
        print("Training Confusion Matrix:\n", train_cm)
        print("Test Confusion Matrix:\n", test_cm)
        print(f"Training Precision: {train_precision}, Test Precision: {test_precision}")
        print(f"Training Recall: {train_recall}, Test Recall: {test_recall}")
        print(f"Training F1-Score: {train_f1}, Test F1-Score: {test_f1}")
        print(f"Training Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
        
        # A3: Generate synthetic data and scatter plot
        np.random.seed(42)
        X_synthetic = np.random.randint(1, 10, size=(20, 2))
        y_synthetic = np.random.choice([0, 1], size=20)
        
        plt.scatter(X_synthetic[:, 0], X_synthetic[:, 1], c=['blue' if label == 0 else 'red' for label in y_synthetic])
        plt.title("Synthetic Training Data")
        plt.xlabel("Feature X")
        plt.ylabel("Feature Y")
        plt.show()
        
        # A4: Classify test points with kNN (k=3)
        x_range = np.arange(0, 10.1, 0.1)
        y_range = np.arange(0, 10.1, 0.1)
        xx, yy = np.meshgrid(x_range, y_range)
        X_test_synthetic = np.c_[xx.ravel(), yy.ravel()]
        
        knn_synthetic = KNeighborsClassifier(n_neighbors=3)
        knn_synthetic.fit(X_synthetic, y_synthetic)
        y_test_pred_synthetic = knn_synthetic.predict(X_test_synthetic)
        
        plt.scatter(X_test_synthetic[:, 0], X_test_synthetic[:, 1], c=['blue' if label == 0 else 'red' for label in y_test_pred_synthetic], alpha=0.4)
        plt.title("kNN Classification (k=3)")
        plt.xlabel("Feature X")
        plt.ylabel("Feature Y")
        plt.show()
        
        # A5: Repeat for different k values
        for k in [1, 5, 10]:
            plot_decision_boundary(X_synthetic, y_synthetic, X_test_synthetic, y_test_pred_synthetic, k)
        
        # A6: Repeat for project data
        plot_decision_boundary(X_train[:, :2], y_train, X_test[:, :2], y_test, k=3)
        
        # A7: Hyperparameter tuning
        best_k = tune_knn_hyperparameters(X_train, y_train)
        print(f"Ideal k value: {best_k}")
