# Credit Risk Scorer

An explainable machine learning web app that predicts loan default risk using XGBoost on the LendingClub dataset. Every prediction comes with a plain-language SHAP explanation so users understand exactly why a loan was flagged as high risk.

---

## Demo

| Input Form | Prediction + SHAP |
|------------|-------------------|
| Loan details, borrower profile, credit history | Default probability, risk label, waterfall chart, 3-line plain-English explanation |

---

## Features

- **XGBoost classifier** trained on 200,000 LendingClub loans (2007–2018)
- **SHAP TreeExplainer** — per-prediction waterfall charts showing which features drove the score
- **Plain-English explanations** — top 3 factors stated in plain language
- **Optimal threshold tuning** — F1-maximising threshold found automatically at training time
- **Class imbalance handling** — `scale_pos_weight` set from training data ratio
- **Streamlit UI** — two-column layout with sliders, dropdowns, and live prediction

---

## Model Performance

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.723 |
| F1-Score (Charged Off) | 0.437 |
| Training samples | 200,000 |
| Test samples | 40,000 |
| Default rate | ~19.9% |

> **Note:** The AUC ceiling for pre-origination-only features on this dataset is ~0.72–0.75. Tutorials claiming 0.91+ use post-origination features (total payments received, outstanding principal) that are not available at loan decision time and would constitute data leakage in a real system.

---

## Tech Stack

| Layer | Tool |
|-------|------|
| ML Model | XGBoost 2.x |
| Explainability | SHAP |
| Data | LendingClub (2007–2018Q4) |
| UI | Streamlit |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualisation | Plotly, Matplotlib |
| Hosting | HuggingFace Spaces |

---

## Project Structure

```
credit-risk-scorer/
├── accepted_2007_to_2018Q4.csv   ← LendingClub accepted loans dataset
├── rejected_2007_to_2018Q4.csv   ← LendingClub rejected loans dataset
├── requirements.txt
├── models/
│   └── credit_model.pkl           ← trained model + label encoders + threshold
├── notebooks/
│   ├── 01_eda.ipynb               ← exploratory data analysis
│   ├── 02_feature_eng.ipynb       ← feature engineering walkthrough
│   └── 03_model_train.ipynb       ← training, evaluation, SHAP plots
└── src/
    ├── preprocess.py              ← cleaning, encoding, feature engineering
    ├── train.py                   ← model training script
    ├── explain.py                 ← SHAP utilities + plain-language generator
    └── app.py                     ← Streamlit web app
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> Requires **Python 3.10–3.13**. Python 3.14 is not yet supported by XGBoost/SHAP.

### 2. Train the model

```bash
cd src
python train.py ../accepted_2007_to_2018Q4.csv
```

This samples 200k rows, trains XGBoost with early stopping, tunes the classification threshold, and saves `models/credit_model.pkl`.

### 3. Run the app

```bash
cd src
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Features Used

### Raw features (24)

| Feature | Description |
|---------|-------------|
| `loan_amnt` | Requested loan amount |
| `int_rate` | Interest rate (%) |
| `installment` | Monthly payment ($) |
| `annual_inc` | Annual income |
| `dti` | Debt-to-income ratio |
| `fico_range_low / high` | FICO credit score bounds |
| `delinq_2yrs` | Delinquencies in last 2 years |
| `open_acc` | Open credit lines |
| `pub_rec` | Public derogatory records |
| `revol_bal` | Revolving credit balance |
| `revol_util` | Credit card utilisation (%) |
| `total_acc` | Total credit lines ever |
| `mort_acc` | Mortgage accounts |
| `pub_rec_bankruptcies` | Bankruptcy records |
| `inq_last_6mths` | Credit inquiries in last 6 months |
| `grade` | LendingClub loan grade (A–G, ordinal) |
| `sub_grade` | Sub-grade (A1–G5, ordinal) |
| `emp_length` | Employment length (ordinal) |
| `earliest_cr_line` | Credit history length in years |
| `purpose` | Loan purpose (label encoded) |
| `home_ownership` | Rent / Mortgage / Own (label encoded) |
| `verification_status` | Income verification status |
| `term` | Loan term (36 or 60 months) |

### Engineered features (4)

| Feature | Formula |
|---------|---------|
| `income_to_loan_ratio` | `annual_inc / (loan_amnt + 1)` |
| `credit_utilization_risk` | `revol_util × dti` |
| `monthly_debt_burden` | `installment / (annual_inc/12 + 1)` |
| `fico_range` | `fico_range_high − fico_range_low` |

---

## Notebooks

| Notebook | What it covers |
|----------|---------------|
| `01_eda.ipynb` | Target distribution, missing values, grade vs default rate, FICO bands, correlation heatmap, Plotly charts |
| `02_feature_eng.ipynb` | String cleaning, null handling, ordinal encoding, label encoding, engineered features |
| `03_model_train.ipynb` | XGBoost training, threshold curve, ROC, confusion matrix, SHAP summary + waterfall + bar |

Launch Jupyter:

```bash
pip install jupyter
jupyter notebook
```

---

## HuggingFace Spaces Deployment

```bash
git init
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/credit-risk-scorer
git add src/app.py src/preprocess.py src/explain.py models/credit_model.pkl requirements.txt
git commit -m "deploy"
git push
```

---

## Data

- **Source:** [LendingClub Loan Data on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- **File used:** `accepted_2007_to_2018Q4.csv` (~2.26M rows, 151 columns)
- **Sample used for training:** 200,000 rows (stratified on `loan_status`)
- **Target:** `Charged Off` = 1 (default), `Fully Paid` = 0

---

## License

MIT
