import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report

import numpy as np
from preprocess import preprocess, FEATURES, label_encoders

ENGINEERED = ['income_to_loan_ratio', 'credit_utilization_risk', 'monthly_debt_burden', 'fico_range']
ALL_FEATURES = FEATURES + ENGINEERED
MODEL_PATH = '../models/credit_model.pkl'


def train(csv_path, sample_size=200_000):
    print("Loading and preprocessing data...")
    df = preprocess(csv_path, sample_size)

    X = df[ALL_FEATURES]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Default rate: {y.mean():.2%}")

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        early_stopping_rounds=30,
        random_state=42,
        device='cpu',
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Find threshold that maximises F1 on Charged Off class
    thresholds = np.arange(0.2, 0.7, 0.01)
    best_f1, best_thresh = 0.0, 0.45
    for t in thresholds:
        f = f1_score(y_test, (y_pred_proba >= t).astype(int))
        if f > best_f1:
            best_f1, best_thresh = f, t
    print(f"Best threshold: {best_thresh:.2f}")
    y_pred = (y_pred_proba >= best_thresh).astype(int)

    print(f"\nAUC-ROC : {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print("\n", classification_report(y_test, y_pred, target_names=['Fully Paid', 'Charged Off']))

    joblib.dump({'model': model, 'features': ALL_FEATURES, 'label_encoders': label_encoders, 'threshold': best_thresh}, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    return model, X_test, y_test


if __name__ == '__main__':
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else '../accepted_2007_to_2018Q4.csv'
    train(csv)
