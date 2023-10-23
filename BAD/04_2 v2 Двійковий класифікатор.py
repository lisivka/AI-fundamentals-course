import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier


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

def preprocess_text_data(dataset, target_column, vectorizer=None):
    """
    Preprocess text data for machine learning tasks.

    This function takes a dataset containing text features and their corresponding
    target values, and preprocesses the text data using the TF-IDF vectorization technique.
    It returns the transformed text features, target values, and the vectorizer used
    for transformation.

    Args:
        dataset (pandas.DataFrame): The dataset containing 'Text' and 'target_column'.
        target_column (str): The name of the column containing target values.
        vectorizer (sklearn.feature_extraction.text.TfidfVectorizer, optional):
            An optional TF-IDF vectorizer. If None, a new vectorizer will be created.

    Returns:
        tuple: A tuple containing:
            - text_features_transformed (scipy.sparse.csr.csr_matrix):
              Transformed text features using TF-IDF vectorization.
            - target (pandas.Series): Target values from the specified target_column.
            - vectorizer (sklearn.feature_extraction.text.TfidfVectorizer):
              The TF-IDF vectorizer used for transformation.
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer()
    text_features_transformed = vectorizer.fit_transform(dataset['Text'])
    target = dataset[target_column]
    # print(f"text_features_transformed: {text_features_transformed}")
    # print(f"target: {target}")
    # print(f"vectorizer: {vectorizer}")


    return text_features_transformed, target, vectorizer

def train_classifier(x_train, y_train):
    """
    Train a classifier using the provided training data.

    This function takes training data in the form of feature matrix 'X_train' and
    corresponding target values 'y_train', and trains a logistic regression classifier.
    The trained classifier is returned for further use.

    Args:
        x_train (scipy.sparse.csr.csr_matrix or array-like): Training feature matrix.
        y_train (array-like): Target values for training.

    Returns:
        sklearn.linear_model.LogisticRegression: Trained logistic regression classifier.
    """
    classifier = LogisticRegression()
    classifier.fit(x_train, y_train)

    return classifier

def evaluate_classifier(classifier, x_val, y_val):
    """
    Evaluate the performance of a classifier on validation data.

    This function takes a trained classifier, validation feature matrix 'x_val', and
    corresponding validation target values 'y_val'. It calculates and returns various
    performance metrics including accuracy, precision, recall, F1 score, and ROC AUC score.

    Args:
        classifier (sklearn.base.BaseEstimator): Trained classifier to be evaluated.
        x_val (scipy.sparse.csr.csr_matrix or array-like): Validation feature matrix.
        y_val (array-like): Validation target values.

    Returns:
        dict: A dictionary containing performance metrics:
              - 'accuracy': Accuracy score.
              - 'precision': Precision score.
              - 'recall': Recall score.
              - 'f1_score': F1 score.
              - 'roc_auc': ROC AUC score.
    """
    y_pred = classifier.predict(x_val)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, classifier.predict_proba(x_val)[:, 1])

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

    return metrics



# Load the CyberBullyingTweets dataset
file_path = '../CyberBullying_Comments_Dataset.csv'
target_column = 'CB_Label'  # Replace with the actual label column name
data = load_dataset(file_path)

# Preprocess the text data and extract features
x_text, y, text_vectorizer = preprocess_text_data(data, target_column)
# print(x_text, y, text_vectorizer)

# Split the data into train and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_text, y, test_size=0.2, random_state=42)

# print(f"x_train: {x_train}")
# print(f"x_val: {x_val}")
# print(f"y_train: {y_train}")
# print(f"y_val: {y_val}")

# Train the logistic regression classifier
classifier = train_classifier(x_train, y_train)

# Evaluate the classifier
evaluation_metrics = evaluate_classifier(classifier, x_val, y_val)
print(evaluation_metrics, sep="\n")



X_train, X_test, y_train, y_test = train_test_split(x_text, y, test_size=0.20,
                                                    random_state=27)

SVC_model = svm.SVC()
KNN_model = KNeighborsClassifier(n_neighbors=5)
SVC_model.fit(x_train, y_train)
KNN_model.fit(x_train, y_train)
SVC_prediction = SVC_model.predict(x_val)
KNN_prediction = KNN_model.predict(x_val)
# print(f"SVC_prediction: {SVC_prediction}")
# print(f"KNN_prediction: {KNN_prediction}")

# Оценка точности — простейший вариант оценки работы классификатора
print(accuracy_score(SVC_prediction, y_test))
print(accuracy_score(KNN_prediction, y_test))
# Но матрица неточности и отчёт о классификации дадут больше информации о производительности

# print(confusion_matrix(SVC_prediction, y_test))
# print(classification_report(KNN_prediction, y_test))


