import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

def preprocess_data(dataset):
    """
    Preprocess the dataset by selecting relevant features and target.

    This function takes a dataset and extracts the 'square_footage' and 'num_bedrooms'
    columns as features, and the 'price' column as the target variable. The preprocessed
    features and target are returned for further use.

    Args:
        dataset (pandas.DataFrame): The dataset containing relevant columns.

    Returns:
        tuple: A tuple containing:
            - X (pandas.DataFrame): Preprocessed feature matrix with 'square_footage' and 'num_bedrooms'.
            - y (pandas.Series): Target values from the 'price' column.
    """
    X = dataset[['square_footage', 'num_bedrooms'],]
    y = dataset['price']
    return X, y

def train_regression_model(X_train, y_train):
    """
    Train a regression model using the provided training data.

    This function takes training data consisting of a feature matrix 'X_train' and
    corresponding target values 'y_train', and trains a linear regression model.
    The trained model is returned for further use.

    Args:
        X_train (array-like): Training feature matrix.
        y_train (array-like): Target values for training.

    Returns:
        sklearn.linear_model.LinearRegression: Trained linear regression model.
    """
    model = LinearRegression( )
    model.fit(X_train, y_train)
    return model

def evaluate_regression_model(model, X_val, y_val):
    """
    Evaluate the performance of a regression model on validation data.

    This function takes a trained regression 'model', validation feature matrix 'X_val',
    and corresponding validation target values 'y_val'. It calculates and returns various
    performance metrics including Mean Squared Error (MSE), Root Mean Squared Error (RMSE),
    Mean Absolute Error (MAE), and R-squared score.

    Args:
        model (sklearn.base.BaseEstimator): Trained regression model to be evaluated.
        X_val (array-like): Validation feature matrix.
        y_val (array-like): Validation target values.

    Returns:
        dict: A dictionary containing performance metrics:
              - 'MSE': Mean Squared Error.
              - 'RMSE': Root Mean Squared Error.
              - 'MAE': Mean Absolute Error.
              - 'R-squared': R-squared score.
    """
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mae = mean_absolute_error(y_val, y_pred)
    r_squared = r2_score(y_val, y_pred)

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R-squared': r_squared}

# Load the dataset
file_path = '../DONE/housing.csv'  # Replace with actual file path
data = load_dataset(file_path)

# Preprocess the data
X, y = preprocess_data(data)

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the regression model
model = train_regression_model(X_train, y_train)

# Evaluate the model
evaluation_metrics = evaluate_regression_model(model, X_val, y_val)
print("Evaluation Metrics:", evaluation_metrics)
