import logging

import pandas as pd
from zenml import step

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoadData:
    """
    Loading the data from the data_path
    """

    def __init__(self, data_path: str):
        """
        Args:
            data_path: path to the data
        """
        self.data_path = data_path

    def get_data(self) -> pd.DataFrame:
        """
        Loading the data from the data_path.

        Returns:
            pd.DataFrame: Loaded data
        """
        logger.info(f"Loading data from {self.data_path}")

        # ✅ بدون parse_dates برای جلوگیری از warning
        df = pd.read_csv(self.data_path)

        logger.info(f"✅ Data Loading successfully. Shape: {df.shape}")
        return df


@step
def load_data(data_path: str) -> pd.DataFrame:
    """
    loading data from data path

    Args:
        data_path: path to the data
    return:
        pd.DataFrame: Loaded data
    """
    try:
        loaded_data = LoadData(data_path)
        data = loaded_data.get_data()
        logger.info(f"Column: {data.columns.tolist()}")
        logger.info(f"Data types:\n{data.dtypes}")
        logger.info(f"Missing values:\n{data.isnull().sum()}")
        return data
    except Exception as e:
        logger.error(f"Error while loading data: {e}", exc_info=True)
        raise e