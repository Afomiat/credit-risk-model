"""
data_processing.py
Feature engineering pipeline for Bati Bank credit risk model.

Transforms raw Xente transaction data into a model-ready
customer-level dataset with engineered features.

Pipeline steps:
1. Aggregate transactions to customer level
2. Extract time-based features
3. Handle missing values
4. Encode categorical variables
5. Scale numerical features

Author: Your Name
Date: May 2026
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import logging
import os

# Set up logging so every step is trackable
# This is important for Basel II audit requirements
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─── Custom Transformers ─────────────────────────────────────
# sklearn pipelines require all steps to be transformers
# We build custom ones for our domain-specific logic

class AggregateFeatures(BaseEstimator, TransformerMixin):
    """
    Aggregates transaction-level data to customer level.

    Transforms 95,662 transaction rows into 3,742 customer
    rows with behavioral features.

    Features created:
    - TotalTransactions: how many times they transacted
    - TotalAmount: sum of all transaction amounts
    - AvgAmount: average transaction amount
    - StdAmount: variability of transaction amounts
    - MaxAmount: largest single transaction
    - MinAmount: smallest single transaction
    - TotalValue: sum of absolute transaction values
    - AvgValue: average absolute transaction value
    - UniqueProducts: diversity of products purchased
    - UniqueProviders: diversity of providers used
    - UniqueChannels: channels used (web, mobile, etc.)
    - FraudCount: number of fraud transactions
    - FraudRate: proportion of transactions that were fraud
    - PositiveAmountCount: number of debit transactions
    - NegativeAmountCount: number of credit transactions
    - PositiveAmountRatio: proportion of debits
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info("Aggregating transactions to customer level...")
        df = X.copy()

        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(
            df['TransactionStartTime']
        ):
            df['TransactionStartTime'] = pd.to_datetime(
                df['TransactionStartTime'], utc=True,
                errors='coerce'
            )

        agg = df.groupby('CustomerId').agg(

            # Transaction volume features
            TotalTransactions=('TransactionId', 'count'),
            UniqueBatches=('BatchId', 'nunique'),

            # Amount features — core financial behavior
            TotalAmount=('Amount', 'sum'),
            AvgAmount=('Amount', 'mean'),
            StdAmount=('Amount', 'std'),
            MaxAmount=('Amount', 'max'),
            MinAmount=('Amount', 'min'),
            MedianAmount=('Amount', 'median'),

            # Value features
            TotalValue=('Value', 'sum'),
            AvgValue=('Value', 'mean'),
            MaxValue=('Value', 'max'),

            # Product diversity
            UniqueProducts=('ProductId', 'nunique'),
            UniqueCategories=('ProductCategory', 'nunique'),
            UniqueProviders=('ProviderId', 'nunique'),
            UniqueChannels=('ChannelId', 'nunique'),

            # Fraud behavior
            FraudCount=('FraudResult', 'sum'),

            # Time features
            FirstTransaction=('TransactionStartTime', 'min'),
            LastTransaction=('TransactionStartTime', 'max'),

            # Positive vs negative amount counts
            PositiveCount=('Amount', lambda x: (x > 0).sum()),
            NegativeCount=('Amount', lambda x: (x < 0).sum()),

        ).reset_index()

        # Derived features
        agg['FraudRate'] = (
            agg['FraudCount'] / agg['TotalTransactions']
        )

        agg['PositiveAmountRatio'] = (
            agg['PositiveCount'] / agg['TotalTransactions']
        )

        # Customer tenure in days
        agg['TenureDays'] = (
            agg['LastTransaction'] - agg['FirstTransaction']
        ).dt.days

        # Transaction velocity (transactions per day)
        # Avoid division by zero for single-day customers
        agg['TransactionVelocity'] = agg['TotalTransactions'] / (
            agg['TenureDays'].replace(0, 1)
        )

        # Fill StdAmount NaN (customers with 1 transaction
        # have no standard deviation)
        agg['StdAmount'] = agg['StdAmount'].fillna(0)

        # Drop raw datetime columns — not needed downstream
        agg = agg.drop(
            columns=['FirstTransaction', 'LastTransaction']
        )

        logger.info(
            f"Aggregation complete: "
            f"{len(agg):,} customers, "
            f"{agg.shape[1]} features"
        )
        return agg


class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts time-based features from TransactionStartTime.

    Applied BEFORE aggregation to get time patterns
    per customer.

    Features extracted:
    - TransactionHour: hour of day (0-23)
    - TransactionDay: day of month (1-31)
    - TransactionMonth: month (1-12)
    - TransactionYear: year
    - TransactionDayOfWeek: 0=Monday, 6=Sunday
    - IsWeekend: 1 if Saturday or Sunday
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info("Extracting time features...")
        df = X.copy()

        if not pd.api.types.is_datetime64_any_dtype(
            df['TransactionStartTime']
        ):
            df['TransactionStartTime'] = pd.to_datetime(
                df['TransactionStartTime'], utc=True,
                errors='coerce'
            )

        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['TransactionDay'] = df['TransactionStartTime'].dt.day
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month
        df['TransactionYear'] = df['TransactionStartTime'].dt.year
        df['TransactionDayOfWeek'] = df['TransactionStartTime'].dt.dayofweek
        df['IsWeekend'] = (
            df['TransactionDayOfWeek'] >= 5
        ).astype(int)

        logger.info("Time features extracted")
        return df


class DropConstantColumns(BaseEstimator, TransformerMixin):
    """
    Drops columns with zero variance (constant values).

    From EDA: CountryCode and CurrencyCode have only
    one unique value each — they provide zero predictive
    signal and waste model capacity.
    """

    def __init__(self):
        self.constant_cols_ = []

    def fit(self, X, y=None):
        # Find columns where all values are the same
        self.constant_cols_ = [
            col for col in X.columns
            if X[col].nunique() <= 1
        ]
        logger.info(
            f"Constant columns to drop: {self.constant_cols_}"
        )
        return self

    def transform(self, X):
        cols_to_drop = [
            c for c in self.constant_cols_
            if c in X.columns
        ]
        return X.drop(columns=cols_to_drop)


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Applies log1p transformation to skewed numerical columns.

    From EDA: Amount has skewness of 51.10.
    Log transformation brings heavily skewed distributions
    closer to normal, improving model performance.

    Uses log1p (log(x+1)) to safely handle zero values.
    For negative values, we transform the absolute value
    and restore the sign.
    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        cols = self.columns or []

        for col in cols:
            if col in df.columns:
                # Handle negative values by preserving sign
                df[col] = np.sign(df[col]) * np.log1p(
                    np.abs(df[col])
                )

        logger.info(f"Log transformation applied to: {cols}")
        return df


# ─── Main Pipeline Builder ───────────────────────────────────

def build_feature_pipeline():
    """
    Builds the full feature engineering pipeline.

    This pipeline takes raw transaction-level data and
    produces a model-ready customer-level dataset.

    Returns:
        sklearn Pipeline object
    """
    pipeline = Pipeline([
        ('time_features', TimeFeatureExtractor()),
        ('drop_constants', DropConstantColumns()),
        ('aggregate', AggregateFeatures()),
    ])

    return pipeline


def build_preprocessing_pipeline(
    numerical_cols: list,
    categorical_cols: list,
    scaling: str = 'standard'
) -> ColumnTransformer:
    """
    Builds the preprocessing pipeline for model-ready data.

    Applies:
    - Median imputation for numerical features
    - Standard or MinMax scaling for numerical features
    - One-hot encoding for categorical features

    Args:
        numerical_cols:   list of numerical column names
        categorical_cols: list of categorical column names
        scaling:          'standard' or 'minmax'

    Returns:
        ColumnTransformer object
    """
    # Choose scaler based on parameter
    scaler = (
        StandardScaler()
        if scaling == 'standard'
        else MinMaxScaler()
    )

    # Numerical pipeline
    # Impute missing → scale
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', scaler),
    ])

    # Categorical pipeline
    # Impute missing → one-hot encode
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False
        )),
    ])

    # Combine both pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols),
        ],
        remainder='drop'  # drop columns not specified
    )

    return preprocessor


def process_raw_data(input_path: str,
                     output_path: str) -> pd.DataFrame:
    """
    Master function: loads raw data, applies feature
    engineering pipeline, saves processed dataset.

    Args:
        input_path:  path to raw data CSV
        output_path: path to save processed CSV

    Returns:
        Processed customer-level DataFrame
    """
    logger.info("=" * 55)
    logger.info("STARTING FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 55)

    # Load raw data
    logger.info(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(
        f"Raw data: {df.shape[0]:,} rows, "
        f"{df.shape[1]} columns"
    )

    # Build and apply pipeline
    pipeline = build_feature_pipeline()
    df_processed = pipeline.fit_transform(df)

    logger.info(
        f"Processed: {df_processed.shape[0]:,} customers, "
        f"{df_processed.shape[1]} features"
    )

    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    logger.info(f"Saved to: {output_path}")
    logger.info("=" * 55)

    return df_processed


if __name__ == "__main__":
    process_raw_data(
        input_path='data/raw/data.csv',
        output_path='data/processed/processed_data.csv'
    )