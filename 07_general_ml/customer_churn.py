"""Simple Telco customer churn classifier using LightGBM.

This script downloads the OpenML dataset (id=44228), does minimal
preprocessing, trains a LightGBM classifier, and reports metrics.

Usage:
    python customer_churn.py

The script is intentionally small and readable so it can be extended.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb



@dataclass
class Config:
    data_id: int = 44228
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 200
    early_stopping_rounds: int = 20


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_openml(data_id: int) -> pd.DataFrame:
    """Download dataset from OpenML and return a single DataFrame."""
    logging.info("Fetching dataset id=%s from OpenML", data_id)
    dataset = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    return dataset.frame


def clean_data(df: pd.DataFrame) -> None:
    """Clean and normalize data in-place."""
    # Normalize string columns: strip whitespace and replace empty with NaN
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip().replace("", np.nan)


def fill_missing_values(df: pd.DataFrame) -> None:
    """Fill missing values for numeric and categorical columns in-place."""
    # Fill numerical missing values with the median
    for col in df.select_dtypes(include=["number"]).columns:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)
    # Fill categorical missing values
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if df[col].isna().any():
            df[col].fillna("__missing__", inplace=True)


def encode_categorical_features(df: pd.DataFrame) -> None:
    """Encode categorical features to 'category' dtype for LightGBM in-place."""
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category")


def encode_target(target_series: pd.Series) -> pd.Series:
    """Encode the target variable into numerical format."""
    if target_series.dtype == "object" or target_series.dtype.name == "category":
        le = LabelEncoder()
        return pd.Series(le.fit_transform(target_series), name=target_series.name)
    return target_series


def split_data(
    X: pd.DataFrame, y: pd.Series, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into training and testing sets."""
    logging.info("Splitting data (test_size=%.3f)", cfg.test_size)
    return train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_state
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a LightGBM churn classifier (OpenML id=44228)")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=200)
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = Config(
        data_id=44228,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
    )

    # 1. Load and preprocess data
    df = load_openml(cfg.data_id)
    clean_data(df)
    fill_missing_values(df)

    # 2. Separate features and target
    target_col = "CHURN"
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # 3. Encode features and target
    encode_categorical_features(X)
    y = encode_target(y)

    # 4. Split data
    X_train, X_test, y_train, y_test = split_data(X, y, cfg)

    # 5. Train model
    categorical_features = [c for c in X_train.columns if X_train[c].dtype.name == "category"]

    model = lgb.LGBMClassifier(n_estimators=cfg.n_estimators, random_state=cfg.random_state)
    logging.info("Training LightGBM model (n_estimators=%d)", cfg.n_estimators)
    model.fit(
        X_train,
        y_train,
        categorical_feature=categorical_features or "auto",
    )

    # 6. Evaluate model
    logging.info("Evaluating model on test set")
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, preds)
    logging.info("Accuracy: %.4f", accuracy)
    auc = roc_auc_score(y_test, proba)
    logging.info("ROC AUC: %.4f", auc)
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    main()
