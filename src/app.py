import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from explain import make_explainer, get_shap_values, waterfall_figure, plain_explanation
from preprocess import label_encoders, CAT_COLS

st.set_page_config(page_title="Credit Risk Scorer", page_icon="💳", layout="wide")

@st.cache_resource
def load_model():
    artifact = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'credit_model.pkl'))
    model = artifact['model']
    features = artifact['features']
    explainer = make_explainer(model)
    return model, explainer, features

model, explainer, feature_names = load_model()

st.title("💳 Credit Risk Scorer")
st.caption("Instant loan default risk prediction with plain-language SHAP explanations")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Loan Details")
    loan_amnt = st.slider("Loan Amount ($)", 1000, 40000, 10000, step=500)
    int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 12.0, step=0.5)
    installment = st.number_input("Monthly Installment ($)", value=300.0, min_value=10.0)
    term = st.selectbox("Loan Term", [36, 60], format_func=lambda x: f"{x} months")
    purpose = st.selectbox("Loan Purpose", [
        "debt_consolidation", "credit_card", "home_improvement",
        "major_purchase", "small_business", "car", "medical", "other"
    ])

with col2:
    st.subheader("Borrower Profile")
    annual_inc = st.number_input("Annual Income ($)", value=60000, step=1000)
    dti = st.slider("Debt-to-Income Ratio (%)", 0.0, 50.0, 15.0, step=0.5)
    fico = st.slider("FICO Score", 580, 850, 700)
    home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN"])
    verification_status = st.selectbox("Income Verification", ["Verified", "Source Verified", "Not Verified"])

st.subheader("Credit History")
c1, c2, c3, c4 = st.columns(4)
with c1:
    delinq_2yrs = st.number_input("Delinquencies (2yr)", value=0, min_value=0, max_value=20)
    open_acc = st.number_input("Open Credit Lines", value=8, min_value=0, max_value=50)
with c2:
    pub_rec = st.number_input("Public Records", value=0, min_value=0, max_value=10)
    total_acc = st.number_input("Total Credit Lines", value=20, min_value=0, max_value=100)
with c3:
    revol_bal = st.number_input("Revolving Balance ($)", value=10000, min_value=0)
    revol_util = st.slider("Credit Utilization (%)", 0.0, 100.0, 40.0)
with c4:
    mort_acc = st.number_input("Mortgage Accounts", value=0, min_value=0, max_value=20)
    pub_rec_bankruptcies = st.number_input("Bankruptcies", value=0, min_value=0, max_value=5)

st.divider()

if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
    raw = pd.DataFrame([{
        'loan_amnt': loan_amnt,
        'int_rate': int_rate,
        'installment': installment,
        'annual_inc': annual_inc,
        'dti': dti,
        'delinq_2yrs': delinq_2yrs,
        'fico_range_low': fico,
        'open_acc': open_acc,
        'pub_rec': pub_rec,
        'revol_bal': revol_bal,
        'revol_util': revol_util,
        'total_acc': total_acc,
        'mort_acc': mort_acc,
        'pub_rec_bankruptcies': pub_rec_bankruptcies,
        'purpose': purpose,
        'home_ownership': home_ownership,
        'verification_status': verification_status,
        'term': term,
    }])

    # Encode categoricals using fitted encoders
    for col in CAT_COLS:
        if col in label_encoders:
            le = label_encoders[col]
            val = raw[col].astype(str)
            # Handle unseen labels gracefully
            known = set(le.classes_)
            raw[col] = val.apply(lambda x: le.transform([x])[0] if x in known else 0)

    raw['income_to_loan_ratio'] = raw['annual_inc'] / (raw['loan_amnt'] + 1)
    raw['credit_utilization_risk'] = raw['revol_util'] * raw['dti']

    input_df = raw[feature_names]

    prob = model.predict_proba(input_df)[0][1]
    is_high = prob >= 0.45

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.metric("Default Probability", f"{prob:.1%}")
        if is_high:
            st.error("🔴 HIGH RISK — Likely to default")
        else:
            st.success("🟢 LOW RISK — Likely to repay")

        st.subheader("Why this score?")
        shap_vals = get_shap_values(explainer, input_df)
        for line in plain_explanation(shap_vals, feature_names):
            st.markdown(f"• {line}")

    with res_col2:
        st.subheader("SHAP Feature Impact")
        fig = waterfall_figure(explainer, shap_vals, input_df)
        st.pyplot(fig)
        plt.close(fig)
