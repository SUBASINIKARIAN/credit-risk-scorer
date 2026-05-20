import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report

from preprocess import preprocess, FEATURES

ENGINEERED = ['income_to_loan_ratio', 'credit_utilization_risk']
ALL_FEATURES = FEATURES + ENGINEERED
MODEL_PATH = '../models/credit_model.pkl'


def train(csv_path, sample_size=100_000):
    print("Loading and preprocessing data...")
    df = preprocess(csv_path, sample_size)

    X = df[ALL_FEATURES]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Default rate: {y.mean():.2%}")

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='auc',
        early_stopping_rounds=20,
        random_state=42,
        device='cpu',
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    # Use 0.45 threshold for better F1 on imbalanced data
    y_pred = (y_pred_proba >= 0.45).astype(int)

    print(f"\nAUC-ROC : {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print("\n", classification_report(y_test, y_pred, target_names=['Fully Paid', 'Charged Off']))

    joblib.dump({'model': model, 'features': ALL_FEATURES}, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    return model, X_test, y_test


if __name__ == '__main__':
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else '../data/loan.csv'
    train(csv)
