from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    train_pipeline(data_path="./data/Telco-Customer-Churn.csv")