# How to Train & Run

## 1. Install dependencies
```bash
cd credit-scoring
pip install -r requirements.txt
```

## 2. Place dataset
Put the LendingClub CSV (e.g. `loan.csv`) inside the `data/` folder.

## 3. Train the model
```bash
cd src
python train.py ../data/loan.csv
```
This saves `models/credit_model.pkl`.

## 4. Run the Streamlit app
```bash
cd src
streamlit run app.py
```
