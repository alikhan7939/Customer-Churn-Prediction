from zenml import step
from typing import Tuple, Annotated
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data_cleaning import DataCleaning, StandardStrategy, BinaryStrategy, MulticlassStrategy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[np.ndarray, "X_train"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "X_test"],
    Annotated[np.ndarray, "y_test"],
]:
    """Clean data split into train and test
    Args:
        df:input DataFrame from ingest_df step

    Returns:
          Tuple of (X_train, y_train, X_test, y_test) as numpy arrays
    """
    try:
        logger.info(f"Cleaning data from {df.shape[0]} rows")
        logger.info(f"Cleaning data from {df.shape[0]} columns")


        strategy = StandardStrategy()
        binary_strategy = BinaryStrategy()
        multiclass_strategy = MulticlassStrategy()
        data_cleaning = DataCleaning(data=df,
                                     strategy=strategy,
                                     binary_strategy=binary_strategy,
                                     multiclass_strategy=multiclass_strategy)
        processed_df = data_cleaning.handle_data()

        logger.info(f"Cleaning data from {processed_df.shape[0]} rows")
        logger.info(f"Columns: {processed_df.columns.to_list()}")

        # ✅ Handle categorical columns
        categorical_cols = processed_df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            if col in processed_df.columns:
                try:
                    processed_df[col] = pd.factorize(processed_df[col].to_numpy())[0]
                    logger.info(f"Encoded categorical column: {col}")
                except Exception as e:
                    logger.warning(f"Could not encode {col}: {str(e)}")
                    # Drop the column if encoding fails
                    processed_df = processed_df.drop(col, axis=1)

        # ✅ Prepare X and y
        if processed_df.shape[1] < 2:
            logger.error("DataFrame must have at least 2 columns (features + target)")
            raise ValueError("Not enough columns in processed data")


        X = processed_df.drop(columns=['Churn'])
        y = processed_df['Churn'].astype(int)

        logger.info(f"X dtype: {X.dtypes}, X shape: {X.shape}")
        logger.info(f"y dtype: {y.dtypes}, y shape: {y.shape}")


        # ✅ Ensure y is 1D
        if y.ndim > 1:
            y = y.ravel()

        # ✅ Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        X_train = np.asarray(X_train, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32)

        logger.info(f"✅ Splitting completed:")
        logger.info(f"   X_train: dtype={X_train.dtype}, shape={X_train.shape}")
        logger.info(f"   X_test: dtype={X_test.dtype}, shape={X_test.shape}")
        logger.info(f"   y_train: dtype={y_train.dtype}, shape={y_train.shape}")
        logger.info(f"   y_test: dtype={y_test.dtype}, shape={y_test.shape}")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error(f"❌ Error in clean_df: {str(e)}", exc_info=True)
        raise e