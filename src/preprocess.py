import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

FEATURES = [
    'loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
    'delinq_2yrs', 'fico_range_low', 'open_acc', 'pub_rec',
    'revol_bal', 'revol_util', 'total_acc', 'mort_acc',
    'pub_rec_bankruptcies', 'purpose', 'home_ownership',
    'verification_status', 'term',
]

CAT_COLS = ['purpose', 'home_ownership', 'verification_status']

label_encoders = {}


def load_and_sample(csv_path, sample_size=100_000, random_state=42):
    df = pd.read_csv(csv_path, low_memory=False)
    # Keep only Fully Paid and Charged Off
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=random_state)
    return df.reset_index(drop=True)


def clean(df):
    df = df[FEATURES + ['loan_status']].copy()

    # Clean string columns
    df['int_rate'] = df['int_rate'].astype(str).str.replace('%', '').astype(float)
    df['term'] = df['term'].astype(str).str.strip().str.replace(' months', '').astype(int)
    df['revol_util'] = df['revol_util'].astype(str).str.replace('%', '')
    df['revol_util'] = pd.to_numeric(df['revol_util'], errors='coerce')

    # Fill nulls
    df['mort_acc'] = df['mort_acc'].fillna(0)
    df['pub_rec_bankruptcies'] = df['pub_rec_bankruptcies'].fillna(0)
    df.dropna(inplace=True)

    return df


def encode_categoricals(df, fit=True):
    for col in CAT_COLS:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            df[col] = le.transform(df[col].astype(str))
    return df


def engineer_features(df):
    df['income_to_loan_ratio'] = df['annual_inc'] / (df['loan_amnt'] + 1)
    df['credit_utilization_risk'] = df['revol_util'] * df['dti']
    return df


def make_target(df):
    df['target'] = (df['loan_status'] == 'Charged Off').astype(int)
    return df


def preprocess(csv_path, sample_size=100_000):
    df = load_and_sample(csv_path, sample_size)
    df = clean(df)
    df = encode_categoricals(df, fit=True)
    df = engineer_features(df)
    df = make_target(df)
    return df
