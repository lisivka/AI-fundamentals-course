import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset(file_path):
    """
    Load the dataset from the specified file path.

    Parameters:
    file_path (str): Path to the dataset file.

    Returns:
    pd.DataFrame: Loaded dataset as a Pandas DataFrame.
    """
    dataset = pd.read_csv(file_path)
    return dataset


def inspect_data(dataset, num_rows=5):
    """
    Display the first few rows of the dataset.

    Parameters:
    dataset (pd.DataFrame): Loaded dataset.
    num_rows (int): Number of rows to display.
    """
    head = dataset.head(num_rows)
    print(head)
    print(dataset.info())
    return


def calculate_summary_statistics(dataset):
    """
    Calculate summary statistics for numerical columns in the dataset.
    Parameters:
    dataset (pd.DataFrame): Loaded dataset.
    Returns:
    pd.DataFrame: Summary statistics.
    """
    summary_stats = dataset.describe( )
    print(summary_stats)
    return summary_stats


def visualize_data(dataset):
    """
    Create visualizations to explore the dataset.

    Parameters:
    dataset (pd.DataFrame): Loaded dataset.
    """
    # Example: Histograms for numerical columns
    numerical_columns = dataset.select_dtypes(include=['int64', 'float64'])
    for column in numerical_columns.columns[:2]:
        plt.figure(figsize=(8, 6))
        sns.histplot(dataset[column], kde=True)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

    # Example: Correlation heatmap
    correlation_matrix = dataset.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()

    return None


def handle_missing_values(dataset):
    """
    Handle missing values in the dataset.

    Parameters:
    dataset (pd.DataFrame): Loaded dataset.

    Returns:
    pd.DataFrame: Dataset with missing values handled.
    """
    # Example: Fill missing values with column mean
    dataset_filled = dataset.fillna(
        dataset.mean(numeric_only=True, skipna=True))

    return dataset_filled


# Load the dataset
file_path = 'housing.csv'  # Replace with the actual file path
data = load_dataset(file_path)

# Perform data exploration
inspect_data(data)
summary_stats = calculate_summary_statistics(data)
visualize_data(data)

data_filled = handle_missing_values(data)
# print(data_filled.isnull().sum())
summary_stats = calculate_summary_statistics(data)

