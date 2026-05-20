import shap
import matplotlib.pyplot as plt
import numpy as np


def make_explainer(model):
    return shap.TreeExplainer(model)


def get_shap_values(explainer, input_df):
    return explainer.shap_values(input_df)


def waterfall_figure(explainer, shap_vals, input_df):
    explanation = shap.Explanation(
        values=shap_vals[0],
        base_values=explainer.expected_value,
        data=input_df.iloc[0],
        feature_names=input_df.columns.tolist(),
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    return fig


FEATURE_LABELS = {
    'loan_amnt': 'loan amount',
    'int_rate': 'interest rate',
    'annual_inc': 'annual income',
    'dti': 'debt-to-income ratio',
    'fico_range_low': 'FICO credit score',
    'revol_util': 'credit card utilization',
    'delinq_2yrs': 'recent delinquencies',
    'pub_rec': 'public records',
    'open_acc': 'open credit lines',
    'income_to_loan_ratio': 'income-to-loan ratio',
    'credit_utilization_risk': 'combined credit risk',
    'installment': 'monthly installment',
    'revol_bal': 'revolving balance',
    'total_acc': 'total credit accounts',
    'mort_acc': 'mortgage accounts',
    'pub_rec_bankruptcies': 'bankruptcy records',
    'purpose': 'loan purpose',
    'home_ownership': 'home ownership status',
    'verification_status': 'income verification',
    'term': 'loan term',
}


def plain_explanation(shap_vals, feature_names, top_n=3):
    pairs = sorted(zip(feature_names, shap_vals[0]), key=lambda x: abs(x[1]), reverse=True)
    lines = []
    for feat, val in pairs[:top_n]:
        label = FEATURE_LABELS.get(feat, feat.replace('_', ' '))
        direction = "increases" if val > 0 else "decreases"
        lines.append(f"Your **{label}** {direction} your default risk.")
    return lines
