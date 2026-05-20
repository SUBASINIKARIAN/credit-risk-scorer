import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

FEATURES = [
    'loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
    'delinq_2yrs', 'fico_range_low', 'fico_range_high', 'open_acc', 'pub_rec',
    'revol_bal', 'revol_util', 'total_acc', 'mort_acc',
    'pub_rec_bankruptcies', 'purpose', 'home_ownership',
    'verification_status', 'term',
    'grade', 'sub_grade', 'inq_last_6mths', 'emp_length',
    'earliest_cr_line',
]

CAT_COLS = ['purpose', 'home_ownership', 'verification_status']

GRADE_ORDER = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
SUB_GRADE_ORDER = [f"{g}{n}" for g in GRADE_ORDER for n in range(1, 6)]
EMP_LENGTH_ORDER = [
    '< 1 year', '1 year', '2 years', '3 years', '4 years',
    '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years', 'Unknown',
]

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

    # Parse credit history length in years from earliest_cr_line (e.g. "Jun-1990")
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')
    df['earliest_cr_line'] = (pd.Timestamp('2018-01-01') - df['earliest_cr_line']).dt.days / 365.25

    # Fill nulls
    df['mort_acc'] = df['mort_acc'].fillna(0)
    df['pub_rec_bankruptcies'] = df['pub_rec_bankruptcies'].fillna(0)
    df['inq_last_6mths'] = df['inq_last_6mths'].fillna(0)
    df['emp_length'] = df['emp_length'].fillna('Unknown')
    df['earliest_cr_line'] = df['earliest_cr_line'].fillna(df['earliest_cr_line'].median())
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
            val = df[col].astype(str)
            known = set(le.classes_)
            df[col] = val.apply(lambda x: le.transform([x])[0] if x in known else 0)

    # Ordinal encode grade, sub_grade, emp_length to preserve ordering
    grade_map = {g: i for i, g in enumerate(GRADE_ORDER)}
    sub_grade_map = {sg: i for i, sg in enumerate(SUB_GRADE_ORDER)}
    emp_map = {e: i for i, e in enumerate(EMP_LENGTH_ORDER)}

    df['grade'] = df['grade'].astype(str).map(grade_map).fillna(len(GRADE_ORDER) - 1).astype(int)
    df['sub_grade'] = df['sub_grade'].astype(str).map(sub_grade_map).fillna(len(SUB_GRADE_ORDER) - 1).astype(int)
    df['emp_length'] = df['emp_length'].astype(str).map(emp_map).fillna(len(EMP_LENGTH_ORDER) - 1).astype(int)

    return df


def engineer_features(df):
    df['income_to_loan_ratio'] = df['annual_inc'] / (df['loan_amnt'] + 1)
    df['credit_utilization_risk'] = df['revol_util'] * df['dti']
    df['monthly_debt_burden'] = df['installment'] / (df['annual_inc'] / 12 + 1)
    df['fico_range'] = df['fico_range_high'] - df['fico_range_low']
    return df


def make_target(df):
    df['target'] = (df['loan_status'] == 'Charged Off').astype(int)
    return df


def preprocess(csv_path, sample_size=200_000):
    df = load_and_sample(csv_path, sample_size)
    df = clean(df)
    df = encode_categoricals(df, fit=True)
    df = engineer_features(df)
    df = make_target(df)
    return df
