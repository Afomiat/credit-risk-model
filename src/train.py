"""
train.py
Model training and MLflow experiment tracking
for Bati Bank credit risk model.

Models trained:
- Logistic Regression (interpretable baseline)
- Random Forest (ensemble)
- XGBoost (gradient boosting)

All experiments logged to MLflow.
Best model registered in MLflow Model Registry.

Author: Your Name
Date: May 2026
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import logging
import os
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─── Feature Selection ───────────────────────────────────────

# Columns to exclude from model training
# IDs: not predictive
# Cluster: derived from target — data leakage
EXCLUDE_COLS = [
    'CustomerId', 'Cluster'
]

# Target column
TARGET = 'is_high_risk'


def load_data(filepath: str) -> pd.DataFrame:
    """Load the final processed dataset."""
    logger.info(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(
        f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns"
    )
    logger.info(
        f"Target distribution:\n"
        f"{df[TARGET].value_counts().to_string()}"
    )
    return df


def prepare_data(df: pd.DataFrame) -> tuple:
    """
    Split into features and target.
    Drop excluded columns.
    Split into train/test (80/20).

    Stratify ensures both splits have the same
    proportion of high-risk customers.
    """
    # Drop excluded columns
    drop_cols = [c for c in EXCLUDE_COLS if c in df.columns]
    X = df.drop(columns=drop_cols + [TARGET])
    y = df[TARGET]

    logger.info(f"Features: {X.shape[1]}")
    logger.info(f"Target: {y.value_counts().to_dict()}")

    # 80/20 split with stratification
    # stratify=y ensures both sets have same class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    logger.info(
        f"Train high-risk: {y_train.sum()} "
        f"({y_train.mean():.1%})"
    )
    logger.info(
        f"Test high-risk:  {y_test.sum()} "
        f"({y_test.mean():.1%})"
    )

    return X_train, X_test, y_train, y_test


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str
) -> dict:
    """
    Evaluate a trained model on the test set.
    Returns all metrics as a dictionary.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(
            y_test, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(
            y_test, y_pred, zero_division=0), 4),
        'f1':        round(f1_score(
            y_test, y_pred, zero_division=0), 4),
        'roc_auc':   round(roc_auc_score(y_test, y_prob), 4),
    }

    logger.info(f"\n{model_name} Results:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    logger.info(
        f"\nClassification Report:\n"
        f"{classification_report(y_test, y_pred, zero_division=0)}"
    )

    return metrics


# ─── Model Training Functions ────────────────────────────────

def train_logistic_regression(
    X_train, y_train, X_test, y_test
) -> dict:
    """
    Train Logistic Regression with preprocessing pipeline.

    Why Logistic Regression first:
    - Most interpretable model — coefficients explain
      each feature's contribution to risk
    - Direct Basel II compliance
    - Fast to train — good baseline
    """
    logger.info("\n" + "=" * 55)
    logger.info("Training Logistic Regression...")

    with mlflow.start_run(run_name="LogisticRegression"):

        # Pipeline: impute missing → scale → classify
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
            ('model',   LogisticRegression(
                class_weight='balanced',
                # balanced adjusts for 97/3 imbalance
                max_iter=1000,
                random_state=42,
                C=1.0
            ))
        ])

        params = {
            'model': 'LogisticRegression',
            'C': 1.0,
            'class_weight': 'balanced',
            'max_iter': 1000,
            'random_state': 42
        }

        # Log parameters to MLflow
        mlflow.log_params(params)

        # Train
        pipeline.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(
            pipeline, X_test, y_test,
            "Logistic Regression"
        )

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)

        # Save model artifact to MLflow
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name="CreditRisk_LogisticRegression"
        )

        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run ID: {run_id}")

    return {'model': pipeline, 'metrics': metrics,
            'name': 'LogisticRegression', 'run_id': run_id}


def train_random_forest(
    X_train, y_train, X_test, y_test
) -> dict:
    """
    Train Random Forest classifier.

    Why Random Forest:
    - Handles non-linear relationships
    - Built-in feature importance
    - Robust to outliers (our data has extreme values)
    - Less prone to overfitting than single trees
    """
    logger.info("\n" + "=" * 55)
    logger.info("Training Random Forest...")

    with mlflow.start_run(run_name="RandomForest"):

        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model',   RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ))
        ])

        params = {
            'model': 'RandomForest',
            'n_estimators': 100,
            'max_depth': 10,
            'class_weight': 'balanced',
            'random_state': 42
        }

        mlflow.log_params(params)
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(
            pipeline, X_test, y_test,
            "Random Forest"
        )

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name="CreditRisk_RandomForest"
        )

        run_id = mlflow.active_run().info.run_id

    return {'model': pipeline, 'metrics': metrics,
            'name': 'RandomForest', 'run_id': run_id}


def train_xgboost(
    X_train, y_train, X_test, y_test
) -> dict:
    """
    Train XGBoost classifier.

    Why XGBoost:
    - State-of-the-art on tabular financial data
    - Handles missing values natively
    - scale_pos_weight handles class imbalance
    - Best expected performance (highest AUC)
    """
    logger.info("\n" + "=" * 55)
    logger.info("Training XGBoost...")

    # Class imbalance weight
    # scale_pos_weight = negative/positive
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / pos
    logger.info(
        f"Class balance — 0: {neg}, 1: {pos}, "
        f"scale_pos_weight: {scale:.1f}"
    )

    with mlflow.start_run(run_name="XGBoost"):

        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model',   xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale,
                random_state=42,
                verbosity=0,
                eval_metric='logloss'
            ))
        ])

        params = {
            'model': 'XGBoost',
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'scale_pos_weight': round(scale, 2),
            'random_state': 42
        }

        mlflow.log_params(params)
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(
            pipeline, X_test, y_test,
            "XGBoost"
        )

        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(
            pipeline['model'],
            artifact_path="model",
            registered_model_name="CreditRisk_XGBoost"
        )

        run_id = mlflow.active_run().info.run_id

    return {'model': pipeline, 'metrics': metrics,
            'name': 'XGBoost', 'run_id': run_id}


# ─── Comparison & Summary ────────────────────────────────────

def compare_models(results: list) -> pd.DataFrame:
    """Build a comparison table of all model results."""
    rows = []
    for r in results:
        row = {'Model': r['name']}
        row.update(r['metrics'])
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values('roc_auc', ascending=False)
    return df


def find_best_model(results: list) -> dict:
    """Identify best model by ROC-AUC score."""
    return max(results, key=lambda r: r['metrics']['roc_auc'])


# ─── Main Training Pipeline ──────────────────────────────────

def main():
    """
    Full training pipeline:
    1. Load final dataset
    2. Prepare features and split
    3. Train all three models with MLflow tracking
    4. Compare results
    5. Identify and log best model
    """
    logger.info("=" * 55)
    logger.info("STARTING MODEL TRAINING PIPELINE")
    logger.info("=" * 55)

    # Set MLflow tracking URI
    # MLflow 3.x requires a database backend (file-store removed)
    # sqlite:///mlflow.db creates a local mlflow.db file in project root
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("credit-risk-bati-bank")

    # Load data
    df = load_data('data/processed/final_dataset.csv')

    # Prepare
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Train all models
    results = []

    lr_result = train_logistic_regression(
        X_train, y_train, X_test, y_test
    )
    results.append(lr_result)

    rf_result = train_random_forest(
        X_train, y_train, X_test, y_test
    )
    results.append(rf_result)

    xgb_result = train_xgboost(
        X_train, y_train, X_test, y_test
    )
    results.append(xgb_result)

    # Compare
    comparison = compare_models(results)

    logger.info("\n" + "=" * 55)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 55)
    logger.info(f"\n{comparison.to_string(index=False)}")

    # Best model
    best = find_best_model(results)
    logger.info(f"\n🏆 Best Model: {best['name']}")
    logger.info(f"   ROC-AUC: {best['metrics']['roc_auc']:.4f}")
    logger.info(f"   F1:      {best['metrics']['f1']:.4f}")
    logger.info(f"   Recall:  {best['metrics']['recall']:.4f}")

    # Save comparison to file
    os.makedirs('reports', exist_ok=True)
    comparison.to_csv(
        'reports/model_comparison.csv', index=False
    )
    logger.info(
        "\nModel comparison saved to: "
        "reports/model_comparison.csv"
    )

    logger.info("=" * 55)
    logger.info("TRAINING COMPLETE — check MLflow UI:")
    logger.info("  Run: mlflow ui")
    logger.info("  Open: http://localhost:5000")
    logger.info("=" * 55)

    return results, comparison


if __name__ == "__main__":
    main()