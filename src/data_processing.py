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
    RobustScaler,
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
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


# ─── RFM & Target Variable ───────────────────────────────────


def calculate_rfm(df_raw: pd.DataFrame,
                  snapshot_date: str = None) -> pd.DataFrame:
    """
    Calculate RFM metrics for each customer.

    Recency   = days since last transaction
    Frequency = total number of transactions
    Monetary  = total positive amount spent
    """
    logger.info("Calculating RFM metrics...")

    df = df_raw.copy()

    df['TransactionStartTime'] = pd.to_datetime(
        df['TransactionStartTime'], utc=True, errors='coerce'
    )

    if snapshot_date is None:
        snapshot = (
            df['TransactionStartTime'].max()
            + pd.Timedelta(days=1)
        )
    else:
        snapshot = pd.Timestamp(snapshot_date, tz='UTC')

    logger.info(f"Snapshot date: {snapshot.date()}")

    last_txn = df.groupby('CustomerId')[
        'TransactionStartTime'
    ].max().reset_index()
    last_txn.columns = ['CustomerId', 'LastTransaction']
    last_txn['Recency'] = (
        snapshot - last_txn['LastTransaction']
    ).dt.days

    frequency = df.groupby('CustomerId').size().reset_index()
    frequency.columns = ['CustomerId', 'Frequency']

    monetary = df[df['Amount'] > 0].groupby(
        'CustomerId'
    )['Amount'].sum().reset_index()
    monetary.columns = ['CustomerId', 'Monetary']

    rfm = last_txn[['CustomerId', 'Recency']].merge(
        frequency, on='CustomerId'
    ).merge(
        monetary, on='CustomerId', how='left'
    )

    rfm['Monetary'] = rfm['Monetary'].fillna(0)

    logger.info(f"RFM calculated for {len(rfm):,} customers")
    logger.info(
        f"\nRFM Summary:\n"
        f"{rfm[['Recency','Frequency','Monetary']].describe().round(2)}"
    )

    return rfm


def assign_risk_labels(
    rfm: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Cluster customers by RFM and assign is_high_risk label.

    Uses log-transformed Frequency and Monetary before
    scaling to handle extreme outliers (max 4,091
    transactions, max 83M UGX monetary value).

    High-risk = highest recency + lowest frequency
              + lowest monetary.
    """
    logger.info("Clustering customers by RFM profile...")

    rfm = rfm.copy()

    # Log-transform skewed dimensions before clustering
    # This prevents extreme outliers from dominating
    # the distance calculations in K-Means
    rfm['LogFrequency'] = np.log1p(rfm['Frequency'])
    rfm['LogMonetary'] = np.log1p(rfm['Monetary'])

    # Scale using RobustScaler on log-transformed values
    scaler = RobustScaler()
    rfm_scaled = scaler.fit_transform(
        rfm[['Recency', 'LogFrequency', 'LogMonetary']]
    )

    # K-Means with fixed random_state for reproducibility
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Analyze clusters on ORIGINAL (non-log) values
    # so the business interpretation is readable
    cluster_summary = rfm.groupby('Cluster').agg(
        AvgRecency=('Recency', 'mean'),
        AvgFrequency=('Frequency', 'mean'),
        AvgMonetary=('Monetary', 'mean'),
        CustomerCount=('CustomerId', 'count')
    ).round(2)

    logger.info(f"\nCluster Profiles:\n{cluster_summary}")

    # Risk score:
    # Higher recency = worse (more days since last txn)
    # Lower frequency = worse
    # Lower monetary = worse
    # Use log values for scoring too — more balanced
    cluster_summary['AvgLogFreq'] = rfm.groupby(
        'Cluster'
    )['LogFrequency'].mean()
    cluster_summary['AvgLogMon'] = rfm.groupby(
        'Cluster'
    )['LogMonetary'].mean()

    cluster_summary['RiskScore'] = (
        cluster_summary['AvgRecency']
        - cluster_summary['AvgLogFreq'] * 10
        - cluster_summary['AvgLogMon'] * 5
    )

    high_risk_cluster = cluster_summary['RiskScore'].idxmax()
    logger.info(
        f"High-risk cluster identified: "
        f"Cluster {high_risk_cluster}"
    )
    logger.info(
        f"High-risk profile: "
        f"Avg Recency={cluster_summary.loc[high_risk_cluster,'AvgRecency']:.1f} days, "
        f"Avg Frequency={cluster_summary.loc[high_risk_cluster,'AvgFrequency']:.1f} txns, "
        f"Avg Monetary=UGX {cluster_summary.loc[high_risk_cluster,'AvgMonetary']:,.0f}"
    )

    rfm['is_high_risk'] = (
        rfm['Cluster'] == high_risk_cluster
    ).astype(int)

    risk_counts = rfm['is_high_risk'].value_counts()
    logger.info(
        f"\nRisk Label Distribution:\n"
        f"  Low risk  (0): {risk_counts.get(0, 0):,} "
        f"({risk_counts.get(0, 0)/len(rfm):.1%})\n"
        f"  High risk (1): {risk_counts.get(1, 0):,} "
        f"({risk_counts.get(1, 0)/len(rfm):.1%})"
    )

    return rfm


def build_final_dataset(
    input_path: str,
    output_path: str
) -> pd.DataFrame:
    """
    Master function for Task 4:
    Runs feature engineering + RFM + clustering
    and saves the final labeled dataset.
    """
    logger.info("=" * 55)
    logger.info("BUILDING FINAL DATASET WITH TARGET VARIABLE")
    logger.info("=" * 55)

    df_raw = pd.read_csv(input_path)

    # Feature engineering
    pipeline = build_feature_pipeline()
    df_features = pipeline.fit_transform(df_raw)

    # RFM calculation
    rfm = calculate_rfm(df_raw)

    # Risk label assignment
    rfm_labeled = assign_risk_labels(rfm)

    # Merge features + labels
    df_final = df_features.merge(
        rfm_labeled[[
            'CustomerId', 'Recency',
            'Frequency', 'Monetary',
            'Cluster', 'is_high_risk'
        ]],
        on='CustomerId',
        how='left'
    )

    logger.info(
        f"\nFinal dataset: "
        f"{df_final.shape[0]:,} customers × "
        f"{df_final.shape[1]} features"
    )
    logger.info(
        f"High-risk customers: "
        f"{df_final['is_high_risk'].sum():,} "
        f"({df_final['is_high_risk'].mean():.1%})"
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    logger.info(f"Saved to: {output_path}")
    logger.info("=" * 55)

    return df_final


if __name__ == "__main__":
    build_final_dataset(
        input_path='data/raw/data.csv',
        output_path='data/processed/final_dataset.csv'
    )
