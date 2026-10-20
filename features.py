"""One source of truth for raw-data training and Streamlit feature engineering."""
import pandas as pd

RAW_FEATURES = [
    "no_of_dependents", "education", "self_employed", "income_annum",
    "loan_amount", "loan_term", "cibil_score", "residential_assets_value",
    "commercial_assets_value", "luxury_assets_value", "bank_asset_value",
]
FEATURES = RAW_FEATURES + ["total_asset_value", "dti", "atr"]
ASSETS = ["residential_assets_value", "commercial_assets_value",
          "luxury_assets_value", "bank_asset_value"]


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(RAW_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing raw columns: {sorted(missing)}")
    x = frame[RAW_FEATURES].copy()
    for col, mapping in {
        "education": {"Graduate": 1, "Not Graduate": 0},
        "self_employed": {"Yes": 1, "No": 0},
    }.items():
        x[col] = x[col].map(mapping) if x[col].dtype == object else x[col]
    x = x.apply(pd.to_numeric, errors="raise")
    if x.isna().any().any() or (x["income_annum"] <= 0).any() or (x["loan_amount"] <= 0).any():
        raise ValueError("Inputs must be complete and income/loan amount positive")
    x["total_asset_value"] = x[ASSETS].sum(axis=1)
    x["dti"] = x["loan_amount"] / x["income_annum"]
    x["atr"] = x["total_asset_value"] / x["loan_amount"]
    return x[FEATURES]
