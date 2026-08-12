from zenml import pipeline
from steps.preprocess import clean_data
from steps.load_data import load_data


@pipeline
def train_pipeline(data_path: str):
    """
    Complete training pipeline:
    1. load data from CSV
    2. Clean and preprocess data
    3. Train model
    4. Evaluate model

    Args:
        data_path: Path to the input CSV file
    """
    data = load_data(data_path)
    X_train, X_test, y_train, y_test = clean_data(data)

    return X_train, X_test, y_train, y_test