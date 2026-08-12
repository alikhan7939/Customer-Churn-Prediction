import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
#%%
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Strategy(ABC):
    """ Abstract Strategy class for Data Cleaning """
    @abstractmethod
    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """ handle data cleaning"""
        pass

class StandardStrategy(Strategy):
    """ Abstract Strategy class for Data Cleaning """
    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Clean data safely by checking for column existence"""
        try:
            logger.info(f"Strategy data cleaning shape: {df.shape}")
            logger.info(f"Available columns: {df.columns.to_list()}")

            # Copy data to avoid Setting With Copy warning
            data = df.copy()

            # Step1: Drop datetime columns if they exist

            # Step2:  Handle specific columns (only if they exist)

            # Step3: Drop remaining rows with missing values

            initial_rows = len(data)
            data = data.dropna()
            rows_dropped = initial_rows - len(data)

            if rows_dropped > 0:
                logger.info(f"Dropped {rows_dropped} rows")

            # step 4: Remove duplicates

            initials_rows = len(data)
            data = data.drop_duplicates()
            duplicates_dropped = initials_rows - len(data)

            if duplicates_dropped > 0:
                logger.info(f"Dropped {duplicates_dropped} duplicates rows")

            logger.info(f"Data Cleaning completed. Final shape: {data.shape}")
            return data

        except Exception as e:
            logger.error(f"Error in StandardStrategy.handle_data: {str(e)}", exc_info=True)
            raise e


class OutlierStrategy(Strategy):
    """ Abstract Strategy class for Data Cleaning """
    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method """
        try:
            data = df.copy()
            numeric_cols = data.select_dtypes(include=np.number).columns

            logger.info(f"Handleing outlier in columns: {numeric_cols.to_list()}")

            for col in numeric_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1

                # Remove Outlier
                initial_rows = len(data)
                data = data[
                    (data[col] >= Q1 - 1.5 * IQR) &
                    (data[col] <= Q3 + 1.5 * IQR)
                ]
                rows_removed = initial_rows - len(data)

                if rows_removed > 0:
                    logger.info(f"Removed {rows_removed} rows outlier from {col}")

            return data

        except Exception as ex:
            logger.error(f"Error in OutlierStrategy.handle_data: {str(ex)}", exc_info=True)
            raise ex


class BinaryStrategy(Strategy):
    """ Abstract Strategy class for Binary Columns """
    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Make binary columns scaleable"""
        try:
            data = df.copy()
            binary_cols = [col for col in data.columns
                           if
                         set(data[col].dropna().unique()) <= {'Yes', 'No', 'No internet service', 'Female', 'Male'}]
            initial_rows = len(data)
            for col in binary_cols:
                data[col] = data[col].map({'Yes': 1, 'No': 0, 'No internet service': 0, 'Female': 0, 'Male': 1})

            return data
        except Exception as e:
            logger.error(f"Error in BinaryStrategy.handle_data: {str(e)}", exc_info=True)


class MulticlassStrategy(Strategy):
    """ Abstract Strategy class for  Multiclass Columns """
    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Make multiclass columns scaleable using One-Hot Encoder"""
        try:
            data = df.copy()
            one_hot_columns = [col for col in data.select_dtypes(include=['object']).columns
                              if data[col].nunique() == 3]

            encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

            preprocessor = ColumnTransformer(
                transformers=[
                    ('onehot', encoder, one_hot_columns),
                ],
                remainder='passthrough',
            )

            # Fit + Transform
            transformed_data = preprocessor.fit_transform(data)

            # Create Column Names
            one_hot_feature_names = preprocessor.named_transformers_['onehot'].get_feature_names_out(one_hot_columns)
            remainder_columns = [col for col in data.columns if col not in one_hot_columns]
            all_feature_names = list(one_hot_feature_names) + remainder_columns

            # Final DataFrame
            data = pd.DataFrame(transformed_data, columns=all_feture_names, index=data.index)

            return data
        except Exception as e:
            logger.error(f"Error in MulticlassStrategy.handle_data: {str(e)}", exc_info=True)




class DataCleaning:
    """ Main data cleaning class using strategy pattern """
    def __init__(self, data: pd.DataFrame,
                 strategy: Strategy,
                 binary_strategy: BinaryStrategy,
                 multiclass_strategy: MulticlassStrategy):
        """
        Initialize DataCleaning class

        Args:
            data: Input DataFrame
            strategy: Strategy object for cleaning
        """
        self.data = data
        self.strategy = strategy
        self.binary_strategy = binary_strategy
        self.multiclass_strategy = multiclass_strategy

    def handle_data(self) -> pd.DataFrame:
        """Execute data cleaning using the selected strategy"""
        try:
            logger.info(f"Original  data cleaning shape: {self.data.shape}")

            # Apply Strategy
            processed_data = self.strategy.handle_data(self.data)

            logger.info(f"Cleaned data shape: {processed_data.shape}")

            return processed_data

        except Exception as e:
            logger.error(f"Error in DataCleaning.handle_data: {str(e)}", exc_info=True)
            raise e


