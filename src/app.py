import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from explain import make_explainer, get_shap_values, waterfall_figure, plain_explanation
from preprocess import CAT_COLS, GRADE_ORDER, SUB_GRADE_ORDER, EMP_LENGTH_ORDER

st.set_page_config(
    page_title="Credit Risk Scorer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- global ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding: 2rem 3rem 3rem; max-width: 1200px; }

/* ---- hero banner ---- */
.hero {
    background: linear-gradient(135deg, #1a1f5e 0%, #2d3561 50%, #1e3a8a 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    color: white;
}
.hero h1 { font-size: 2.4rem; font-weight: 700; margin: 0 0 0.4rem; }
.hero p  { font-size: 1.05rem; opacity: 0.85; margin: 0; }
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    margin-bottom: 1rem;
    letter-spacing: 0.05em;
}

/* ---- section cards ---- */
.section-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.4rem 1.6rem 1rem;
    margin-bottom: 1.2rem;
}
.section-title {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ---- result cards ---- */
.result-card {
    border-radius: 14px;
    padding: 1.6rem;
    text-align: center;
    margin-bottom: 1rem;
}
.result-high {
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    border: 2px solid #fca5a5;
}
.result-low {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 2px solid #86efac;
}
.result-label { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.3rem; }
.result-prob  { font-size: 3rem; font-weight: 800; line-height: 1; }
.result-sub   { font-size: 0.88rem; opacity: 0.7; margin-top: 0.3rem; }

/* ---- stat pills ---- */
.stat-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 0.6rem 0; }
.stat-pill {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 0.82rem;
    color: #374151;
    flex: 1;
    min-width: 120px;
    text-align: center;
}
.stat-pill b { display: block; font-size: 1.1rem; color: #1e3a8a; }

/* ---- reason bullets ---- */
.reason-item {
    background: white;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 8px 14px;
    margin: 6px 0;
    font-size: 0.9rem;
    color: #374151;
}

/* ---- disclaimer box ---- */
.disclaimer {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.82rem;
    color: #78350f;
    margin-top: 1.5rem;
    line-height: 1.6;
}

/* ---- grade badge ---- */
.grade-badge {
    display: inline-block;
    width: 36px; height: 36px;
    border-radius: 50%;
    line-height: 36px;
    text-align: center;
    font-weight: 700;
    font-size: 1rem;
    color: white;
}

/* ---- divider ---- */
hr { border: none; border-top: 1px solid #e2e8f0; margin: 1.5rem 0; }

/* hide streamlit chrome */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── helpers ───────────────────────────────────────────────────────────────────

def fico_to_grade(fico):
    if fico >= 760: return 'A'
    if fico >= 720: return 'B'
    if fico >= 690: return 'C'
    if fico >= 660: return 'D'
    if fico >= 630: return 'E'
    if fico >= 600: return 'F'
    return 'G'

GRADE_COLORS = {'A':'#16a34a','B':'#65a30d','C':'#ca8a04','D':'#ea580c','E':'#dc2626','F':'#9f1239','G':'#581c87'}

def grade_to_int_rate(grade):
    return {'A':6.5,'B':10.5,'C':14.0,'D':18.5,'E':22.5,'F':26.0,'G':29.0}[grade]

def calc_installment(principal, annual_rate_pct, term_months):
    r = annual_rate_pct / 100 / 12
    return principal * r * (1+r)**term_months / ((1+r)**term_months - 1) if r else principal / term_months

def risk_gauge(prob):
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw=dict(aspect='equal'))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    theta = np.linspace(np.pi, 0, 300)
    zones = [(0.0, 0.35, '#22c55e'), (0.35, 0.60, '#f59e0b'), (0.60, 1.0, '#ef4444')]
    for lo, hi, color in zones:
        t = theta[int(lo*299):int(hi*299)+1]
        ax.plot(np.cos(t), np.sin(t), lw=18, color=color, solid_capstyle='butt', alpha=0.85)

    angle = np.pi * (1 - prob)
    ax.annotate('', xy=(0.62*np.cos(angle), 0.62*np.sin(angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#1e293b', lw=2.5))
    ax.text(0, -0.18, f"{prob:.0%}", ha='center', va='center',
            fontsize=20, fontweight='bold', color='#1e293b')
    ax.text(-0.85, -0.15, 'LOW', ha='center', fontsize=7, color='#22c55e', fontweight='600')
    ax.text(0, 0.72, 'MED', ha='center', fontsize=7, color='#f59e0b', fontweight='600')
    ax.text(0.85, -0.15, 'HIGH', ha='center', fontsize=7, color='#ef4444', fontweight='600')
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.3, 1.1)
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig

@st.cache_resource
def load_model():
    artifact = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'credit_model.pkl'))
    return (artifact['model'], make_explainer(artifact['model']),
            artifact['features'], artifact['label_encoders'], artifact.get('threshold', 0.45))

model, explainer, feature_names, label_encoders, THRESHOLD = load_model()

# ── hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">ML-POWERED · SHAP EXPLAINABILITY</div>
  <h1>💳 Credit Risk Scorer</h1>
  <p>Answer 10 simple questions to see your loan default risk profile and understand the key factors driving your score.</p>
</div>
""", unsafe_allow_html=True)

# ── form ──────────────────────────────────────────────────────────────────────
with st.form("risk_form"):

    # section 1
    st.markdown('<div class="section-card"><div class="section-title">📋 Step 1 — Loan Details</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        loan_amnt = st.slider("Loan Amount ($)", 1000, 40000, 10000, step=500,
                              help="Total amount you want to borrow.")
    with c2:
        purpose = st.selectbox("Loan Purpose",
            ["debt_consolidation","credit_card","home_improvement",
             "major_purchase","small_business","car","medical","other"],
            help="Primary reason for the loan.")
    with c3:
        term = st.selectbox("Repayment Term", [36, 60],
            format_func=lambda x: f"{x} months ({x//12} yr{'s' if x//12>1 else ''})",
            help="36 months = lower total interest. 60 months = lower monthly payment.")
    st.markdown('</div>', unsafe_allow_html=True)

    # section 2
    st.markdown('<div class="section-card"><div class="section-title">💼 Step 2 — Your Finances</div>', unsafe_allow_html=True)
    c4, c5, c6, c7 = st.columns(4)
    with c4:
        annual_inc = st.number_input("Annual Income ($)", value=60000, step=1000, min_value=5000,
                                     help="Gross (pre-tax) yearly income from all sources.")
    with c5:
        monthly_debt = st.number_input("Monthly Debt Payments ($)", value=500, step=50, min_value=0,
                                       help="Total monthly payments on credit cards, car loans, student loans — excluding rent.")
    with c6:
        home_ownership = st.selectbox("Home Ownership", ["RENT","MORTGAGE","OWN"],
                                      help="RENT = you pay rent · MORTGAGE = you have a home loan · OWN = paid off.")
    with c7:
        emp_length = st.selectbox("Time at Current Job",
            ["< 1 year","1 year","2 years","3 years","4 years",
             "5 years","6 years","7 years","8 years","9 years","10+ years"],
            index=6,
            help="Longer employment typically signals more income stability.")
    st.markdown('</div>', unsafe_allow_html=True)

    # section 3
    st.markdown('<div class="section-card"><div class="section-title">📊 Step 3 — Your Credit Profile</div>', unsafe_allow_html=True)
    c8, c9, c10, c11 = st.columns(4)
    with c8:
        fico = st.slider("Credit Score (FICO)", 580, 850, 700,
                         help="580–629 Poor · 630–689 Fair · 690–719 Good · 720–759 Very Good · 760+ Exceptional\n\nCheck free on Credit Karma or your bank app.")
    with c9:
        revol_util = st.slider("Credit Card Utilization (%)", 0.0, 100.0, 30.0, step=1.0,
                               help="% of your total credit card limit currently in use.\nExample: $3,000 balance on $10,000 limit = 30%.\nBelow 30% is healthy.")
    with c10:
        delinq_2yrs = st.number_input("Late Payments (last 2 yrs)", value=0, min_value=0, max_value=10,
                                      help="Number of payments that were 30+ days late in the past 2 years.")
    with c11:
        inq_last_6mths = st.number_input("Credit Checks (last 6 mo)", value=0, min_value=0, max_value=10,
                                         help="Hard inquiries from lenders when you applied for credit recently. Too many in a short period signals risk.")
    st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("🔍 Calculate My Risk Score", use_container_width=True, type="primary")

# ── glossary expander (outside form) ─────────────────────────────────────────
with st.expander("📖 What do these inputs mean?"):
    st.markdown("""
| Input | Plain-English meaning |
|-------|----------------------|
| **Loan Amount** | How much money you want to borrow |
| **Loan Purpose** | Why you need the money |
| **Repayment Term** | How many months you have to pay it back |
| **Annual Income** | Your total yearly earnings before tax |
| **Monthly Debt Payments** | What you already pay every month on existing debts |
| **Home Ownership** | Whether you rent, have a mortgage, or own your home |
| **Time at Current Job** | How long you've been with your current employer |
| **Credit Score (FICO)** | A 300–850 number summarising your credit history — higher is better |
| **Credit Card Utilization** | How much of your credit card limit you are currently using |
| **Late Payments** | How many times you paid a bill 30+ days late in the last 2 years |
| **Credit Checks** | How many times lenders pulled your credit report recently |
""")

# ── results ───────────────────────────────────────────────────────────────────
if submitted:
    grade       = fico_to_grade(fico)
    int_rate    = grade_to_int_rate(grade)
    installment = calc_installment(loan_amnt, int_rate, term)
    dti         = (monthly_debt / (annual_inc / 12)) * 100
    fico_high   = min(fico + 4, 850)
    mort_acc    = 1 if home_ownership == "MORTGAGE" else 0
    revol_bal   = int((revol_util / 100) * 15000)

    grade_map     = {g: i for i, g in enumerate(GRADE_ORDER)}
    sub_grade_map = {sg: i for i, sg in enumerate(SUB_GRADE_ORDER)}
    emp_map       = {e: i for i, e in enumerate(EMP_LENGTH_ORDER)}

    raw = pd.DataFrame([{
        'loan_amnt': loan_amnt, 'int_rate': int_rate,
        'installment': round(installment, 2), 'annual_inc': annual_inc,
        'dti': dti, 'delinq_2yrs': delinq_2yrs,
        'fico_range_low': fico, 'fico_range_high': fico_high,
        'open_acc': 8, 'pub_rec': 0, 'revol_bal': revol_bal,
        'revol_util': revol_util, 'total_acc': 20, 'mort_acc': mort_acc,
        'pub_rec_bankruptcies': 0, 'purpose': purpose,
        'home_ownership': home_ownership, 'verification_status': 'Not Verified',
        'term': term, 'grade': grade_map.get(grade, 6),
        'sub_grade': sub_grade_map.get(grade + '3', 17),
        'inq_last_6mths': inq_last_6mths,
        'emp_length': emp_map.get(emp_length, 11),
        'earliest_cr_line': 10.0,
    }])

    for col in CAT_COLS:
        if col in label_encoders:
            le = label_encoders[col]
            raw[col] = raw[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in set(le.classes_) else 0)

    raw['income_to_loan_ratio']    = raw['annual_inc'] / (raw['loan_amnt'] + 1)
    raw['credit_utilization_risk'] = raw['revol_util'] * raw['dti']
    raw['monthly_debt_burden']     = raw['installment'] / (raw['annual_inc'] / 12 + 1)
    raw['fico_range']              = raw['fico_range_high'] - raw['fico_range_low']

    input_df = raw[feature_names]
    prob     = model.predict_proba(input_df)[0][1]
    is_high  = prob >= THRESHOLD
    shap_vals = get_shap_values(explainer, input_df)
    reasons   = plain_explanation(shap_vals, feature_names)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Your Risk Assessment")

    left, mid, right = st.columns([1.1, 1.2, 1.7])

    # ── col 1: verdict card ───────────────────────────────────────────────────
    with left:
        card_class = "result-high" if is_high else "result-low"
        verdict    = "HIGH RISK" if is_high else "LOW RISK"
        icon       = "🔴" if is_high else "🟢"
        color      = "#dc2626" if is_high else "#16a34a"
        st.markdown(f"""
<div class="result-card {card_class}">
  <div class="result-label">{icon} {verdict}</div>
  <div class="result-prob" style="color:{color}">{prob:.0%}</div>
  <div class="result-sub">estimated default probability</div>
</div>
""", unsafe_allow_html=True)

        gc = GRADE_COLORS.get(grade, '#64748b')
        st.markdown(f"""
<div class="stat-row">
  <div class="stat-pill"><b style="color:{gc}">{grade}</b>Loan Grade</div>
  <div class="stat-pill"><b>{int_rate:.1f}%</b>Est. Rate</div>
  <div class="stat-pill"><b>${installment:,.0f}</b>Monthly</div>
  <div class="stat-pill"><b>{dti:.1f}%</b>DTI Ratio</div>
</div>
""", unsafe_allow_html=True)

    # ── col 2: gauge + reasons ────────────────────────────────────────────────
    with mid:
        fig_gauge = risk_gauge(prob)
        st.pyplot(fig_gauge, use_container_width=True)
        plt.close(fig_gauge)

        st.markdown("**Key factors driving your score:**")
        for r in reasons:
            st.markdown(f'<div class="reason-item">{r}</div>', unsafe_allow_html=True)

    # ── col 3: SHAP waterfall ─────────────────────────────────────────────────
    with right:
        st.markdown("**Feature impact breakdown (SHAP)**")
        st.caption("➡ Right = increases risk &nbsp;|&nbsp; ⬅ Left = decreases risk")
        fig_shap = waterfall_figure(explainer, shap_vals, input_df)
        st.pyplot(fig_shap, use_container_width=True)
        plt.close(fig_shap)

    # ── disclaimer ────────────────────────────────────────────────────────────
    st.markdown("""
<div class="disclaimer">
⚠️ <b>Disclaimer:</b> This score is generated by a machine learning model trained on historical
LendingClub data (2007–2018) and is provided for <b>educational purposes only</b>. It does not
constitute a credit decision, financial advice, or any guarantee of loan approval or rejection.
Actual lender decisions depend on many additional factors. Consult a qualified financial advisor
for personal financial guidance.
</div>
""", unsafe_allow_html=True)

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Model: XGBoost trained on LendingClub 2007–2018 · Explainability: SHAP · For educational use only · Not financial advice")
