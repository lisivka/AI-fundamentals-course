import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_dataset(file_path):
    """
    Load a dataset from a CSV file.

    This function reads a CSV file located at the specified 'file_path' and returns
    the dataset as a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file to be loaded.

    Returns:
        pandas.DataFrame: The loaded dataset as a DataFrame.

    """
    dataset = pd.read_csv(file_path)
    return dataset


def add_category(dataset):
    new_columns = "ocean_category"
    dataset[new_columns] = 0
    dataset.loc[dataset['ocean_proximity'] == 'NEAR BAY', new_columns] = 1
    dataset.loc[dataset['ocean_proximity'] == '<1H OCEAN', new_columns] = 2
    dataset.loc[dataset['ocean_proximity'] == 'NEAR OCEAN', new_columns] = 3
    dataset.loc[dataset['ocean_proximity'] == 'INLAND', new_columns] = 4
    dataset.loc[dataset['ocean_proximity'] == 'INLAND', new_columns] = 5
    return dataset




def preprocess_data(dataset, selected_features):
    """
    Preprocess the dataset by selecting relevant features and target.

    This function takes a dataset and extracts the selected features, scales them using StandardScaler,
    and extracts the target variable. It also handles missing values by imputing them.

    Args:
        dataset (pandas.DataFrame): The dataset containing relevant columns.
        selected_features (list of str): List of feature column names.

    Returns:
        tuple: A tuple containing:
            - X_scaled (pandas.DataFrame): Scaled feature matrix with selected features.
            - y (pandas.Series): Target values.
    """
    # Select the features and target
    X = dataset[selected_features]
    y = dataset['median_house_value']

    # Handle missing values (if any)
    if X.isnull().any().any():
        # You can choose to either impute missing values or drop rows with missing values
        # In this example, we'll impute missing values using the mean value of each column
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y



def apply_kmeans_clustering(X_scaled, num_clusters):
    """
    Apply K-means clustering to scaled data.

    This function applies the K-means clustering algorithm to the scaled feature matrix
    'X_scaled' using the specified number of clusters 'num_clusters'. It returns the
    cluster assignments for each data point and the cluster centers.

    Args:
        X_scaled (array-like): Scaled feature matrix for clustering.
        num_clusters (int): Number of clusters to form.

    Returns:
        tuple: A tuple containing:
            - clusters (array-like): Cluster assignments for each data point.
            - cluster_centers (array-like): Coordinates of cluster centers.

    """
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    return clusters, kmeans.cluster_centers_


def visualize_clusters(X_scaled, clusters, cluster_centers, features = ['feature1', 'feature2']):
    """
    Visualize the clusters in a 2D scatter plot.

    This function visualizes the K-means clusters by creating a 2D scatter plot with
    different colors for each cluster and cluster centers.

    Args:
        X_scaled (array-like): Scaled feature matrix for clustering.
        clusters (array-like): Cluster assignments for each data point.
        cluster_centers (array-like): Coordinates of cluster centers.

    """
    feature1, feature2 = features

    plt.figure(figsize=(8, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', s=50)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200, label='Cluster Centers')
    plt.title('K-Means Clustering')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend()
    plt.show()


def run_clastering(data, selected_features, num_clusters=5):

    # Preprocess the data
    X, y = preprocess_data(data, selected_features)
    # Apply K-Means clustering
    clusters, cluster_centers = apply_kmeans_clustering(X, num_clusters)
    # Visualize the clusters
    visualize_clusters(X, clusters, cluster_centers, selected_features)

if __name__ == "__main__":
    # Load the dataset
    # file_path = 'features.csv'  # Replace with the actual file path
    file_path = './housing.csv'  # Replace with the actual file path
    data = load_dataset(file_path)
    data = add_category(data)

    # Selected features for clustering
    # selected_features = ['feature1', 'feature2']  # Replace with actual feature names
    selected_features = [
        # 'longitude',
        # 'latitude',
        # 'ocean_category',
        # 'housing_median_age',
        # 'population',
        'households',
        # 'median_income',
        'total_rooms',
        # 'total_bedrooms',

    ]  # Replace with actual feature names

    run_clastering(data, selected_features, num_clusters=5)
    # Preprocess the data
    X, y = preprocess_data(data, selected_features)
    # Choose the number of clusters (K)
    num_clusters = 5  # Replace with the desired number of clusters
    # Apply K-Means clustering
    clusters, cluster_centers = apply_kmeans_clustering(X, num_clusters)
    # Visualize the clusters
    visualize_clusters(X, clusters, cluster_centers, selected_features)


    selected_features = [
        'longitude',
        'latitude',
        # 'ocean_category',
        # 'housing_median_age',
        # 'population',
        # 'households',
        # 'median_income',
        # 'total_rooms',
        # 'total_bedrooms',

    ]  # Replace with actual feature names

    run_clastering(data, selected_features, num_clusters=5)

    # Preprocess the data
    X, y = preprocess_data(data, selected_features)
    # Choose the number of clusters (K)
    num_clusters = 5  # Replace with the desired number of clusters
    # Apply K-Means clustering
    clusters, cluster_centers = apply_kmeans_clustering(X, num_clusters)
    # Visualize the clusters
    visualize_clusters(X, clusters, cluster_centers, selected_features)
