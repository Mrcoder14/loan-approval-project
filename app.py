"""Run: python -m streamlit run app.py"""
import json
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from features import FEATURES, make_features

BASE = {
    "Number of dependents": 1, "Education": "Graduate", "Self-employed": "No",
    "Annual income (₹)": 8_000_000, "Loan amount (₹)": 5_000_000,
    "Loan term (years)": 10, "CIBIL score": 750,
    "Residential asset value (₹)": 10_000_000,
    "Commercial asset value (₹)": 5_000_000,
    "Luxury asset value (₹)": 8_000_000,
    "Bank asset value (₹)": 3_000_000,
}
SCENARIOS = [
    ("Strong credit baseline", {}, "Approved"),
    ("Same profile, lower CIBIL score", {"CIBIL score": 450}, "Rejected"),
    ("Strong credit, few assets", {
        "Residential asset value (₹)": 1_000_000,
        "Commercial asset value (₹)": 500_000,
        "Luxury asset value (₹)": 500_000,
        "Bank asset value (₹)": 500_000,
    }, "Rejected"),
    ("Weak credit, short term, high loan/income and assets", {
        "CIBIL score": 450, "Loan term (years)": 2,
        "Annual income (₹)": 1_500_000, "Loan amount (₹)": 6_000_000,
        "Residential asset value (₹)": 15_000_000,
    }, "Approved"),
    ("Same profile, longer loan term", {
        "CIBIL score": 450, "Loan term (years)": 12,
        "Annual income (₹)": 1_500_000, "Loan amount (₹)": 6_000_000,
        "Residential asset value (₹)": 15_000_000,
    }, "Rejected"),
    ("Short term, smaller loan/income ratio", {
        "CIBIL score": 450, "Loan term (years)": 2,
        "Annual income (₹)": 3_000_000, "Loan amount (₹)": 6_000_000,
        "Residential asset value (₹)": 15_000_000,
    }, "Rejected"),
    ("Change education, employment and dependents", {
        "Education": "Not Graduate", "Self-employed": "Yes",
        "Number of dependents": 5,
    }, "Approved"),
]

ROOT = Path(__file__).resolve().parent
HISTORY_DB = Path(os.environ.get("LOAN_HISTORY_DB", ROOT / "history.sqlite3"))


def save_prediction(inputs, derived, label, probability):
    with sqlite3.connect(HISTORY_DB, timeout=10) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                inputs_json TEXT NOT NULL,
                derived_json TEXT NOT NULL,
                prediction TEXT NOT NULL,
                approved_probability REAL NOT NULL
            )
        """)
        connection.execute(
            """INSERT INTO predictions
               (created_at, inputs_json, derived_json, prediction, approved_probability)
               VALUES (?, ?, ?, ?, ?)""",
            (datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d %H:%M:%S IST"),
             json.dumps(inputs), json.dumps(derived), label, float(probability)),
        )


def load_history():
    if not HISTORY_DB.exists():
        return []
    with sqlite3.connect(HISTORY_DB, timeout=10) as connection:
        rows = connection.execute(
            """SELECT id, created_at, inputs_json, derived_json,
                      prediction, approved_probability
               FROM predictions ORDER BY id DESC LIMIT 100"""
        ).fetchall()
    return [
        {"id": record_id, "time": created_at, **json.loads(inputs_json),
         **json.loads(derived_json), "prediction": prediction,
         "approved_probability": probability}
        for record_id, created_at, inputs_json, derived_json, prediction, probability in rows
    ]


st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦")
st.title("Loan approval prediction")
st.info("Automated demo build — click Start automated demo below.")
st.caption("Academic model of historical Approved/Rejected labels. This is not a bank decision or repayment risk estimate.")
model_file = ROOT / "loan_approval_model.joblib"
if not model_file.exists():
    st.error("Model missing. Place the raw Kaggle CSV in data/ and run `python train_model.py` first.")
    st.stop()

@st.cache_resource
def load_model(path):
    return joblib.load(path)

artifact = load_model(str(model_file))
if artifact["feature_order"] != FEATURES:
    st.error("Model feature order does not match this version of the app.")
    st.stop()


DEMO_FIELDS = {
    "Number of dependents": "no_of_dependents",
    "Education": "education",
    "Self-employed": "self_employed",
    "Annual income (₹)": "income_annum",
    "Loan amount (₹)": "loan_amount",
    "Loan term (years)": "loan_term",
    "CIBIL score": "cibil_score",
    "Residential asset value (₹)": "residential_assets_value",
    "Commercial asset value (₹)": "commercial_assets_value",
    "Luxury asset value (₹)": "luxury_assets_value",
    "Bank asset value (₹)": "bank_asset_value",
}


def predict_one(row):
    x = make_features(row)[artifact["feature_order"]]
    pred = int(artifact["model"].predict(x)[0])
    probability = artifact["model"].predict_proba(x)[
        0, list(artifact["model"].classes_).index(1)
    ]
    label = "Approved" if pred == 1 else "Rejected"
    derived = {key: float(x.iloc[0][key]) for key in ("total_asset_value", "dti", "atr")}
    return label, probability, derived, x


def automated_demo():
    st.subheader("Automated scenario demo")
    st.caption("Click Start automated demo. Seven cases appear in order and are saved in history.")
    display = st.empty()
    if st.button("Start automated demo", type="primary"):
        for index, (name, changes, expected) in enumerate(SCENARIOS, start=1):
            visible_inputs = BASE | changes
            raw_inputs = {DEMO_FIELDS[key]: value for key, value in visible_inputs.items()}
            label, probability, derived, _ = predict_one(pd.DataFrame([raw_inputs]))
            save_prediction(raw_inputs, derived, label, probability)
            with display.container():
                st.write(f"**Scenario {index}/{len(SCENARIOS)}: {name}**")
                st.dataframe(pd.DataFrame([{"Input": key, "Value": str(value)}
                                           for key, value in visible_inputs.items()]),
                             hide_index=True)
                st.success(f"Predicted label: {label} (expected: {expected})")
                st.write(f"Model-estimated Approved probability: {probability:.1%}")
                st.write(f"Total assets: ₹{derived['total_asset_value']:,.0f} | "
                         f"DTI: {derived['dti']:.3f} | ATR: {derived['atr']:.3f}")
            if index < len(SCENARIOS):
                time.sleep(2)
        st.caption("Demo complete. All cases are in Prediction history below.")


automated_demo()

with st.form("application"):
    a, b = st.columns(2)
    with a:
        dependents = st.number_input("Number of dependents", min_value=0, max_value=20, value=None, step=1)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"], index=None, placeholder="Select education")
        employed = st.selectbox("Self-employed", ["No", "Yes"], index=None, placeholder="Select an option")
        income = st.number_input("Annual income (₹)", min_value=1, value=None, step=100000)
        amount = st.number_input("Loan amount (₹)", min_value=1, value=None, step=100000)
        term = st.number_input("Loan term (years)", min_value=1, max_value=40, value=None, step=1)
    with b:
        cibil = st.number_input("CIBIL score", min_value=300, max_value=900, value=None, step=1)
        residential = st.number_input("Residential asset value (₹)", min_value=0, value=None, step=100000)
        commercial = st.number_input("Commercial asset value (₹)", min_value=0, value=None, step=100000)
        luxury = st.number_input("Luxury asset value (₹)", min_value=0, value=None, step=100000)
        bank = st.number_input("Bank asset value (₹)", min_value=0, value=None, step=100000)
    submitted = st.form_submit_button("Predict")

if submitted:
    errors = []
    if any(value is None for value in (dependents, education, employed, income, amount,
                                       term, cibil, residential, commercial, luxury, bank)):
        errors.append("Please complete every field before predicting.")
    if errors:
        for error in errors:
            st.error(error)
    else:
        row = pd.DataFrame([{
            "no_of_dependents": dependents, "education": education,
            "self_employed": employed, "income_annum": income,
            "loan_amount": amount, "loan_term": term, "cibil_score": cibil,
            "residential_assets_value": residential, "commercial_assets_value": commercial,
            "luxury_assets_value": luxury, "bank_asset_value": bank,
        }])
        label, probability, derived, x = predict_one(row)
        save_prediction(row.iloc[0].to_dict(), derived, label, probability)
        st.success(f"Predicted label: {label}")
        st.write(f"Model-estimated Approved probability: {probability:.1%}")
        st.caption("The probability is uncalibrated and should not be treated as a guarantee.")
        with st.expander("Calculated features"):
            st.dataframe(x[["total_asset_value", "dti", "atr"]], hide_index=True)

def history_panel():
    st.subheader("Prediction history")
    st.caption("Recent predictions saved on this computer. The newest case appears first.")
    history = load_history()
    if history:
        summary = pd.DataFrame([{
            "Case": record["id"], "Time": record["time"],
            "CIBIL": record["cibil_score"], "Income (₹)": record["income_annum"],
            "Loan (₹)": record["loan_amount"], "Prediction": record["prediction"],
            "Approved probability": f'{record["approved_probability"]:.1%}',
            "DTI": record["dti"], "ATR": record["atr"],
            "Total assets (₹)": record["total_asset_value"],
        } for record in history])
        st.dataframe(summary, hide_index=True)
        chosen = st.selectbox("View full input and output for case", [r["id"] for r in history])
        selected = next(record for record in history if record["id"] == chosen)
        st.table(pd.DataFrame([{"Field": key, "Value": str(value)}
                               for key, value in selected.items()]))
    else:
        st.info("No predictions saved yet. Submit a profile to create the first record.")


history_panel()
