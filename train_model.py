"""Train a raw-input Random Forest using the supplied train/test membership."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from compare_models import score
from features import FEATURES, RAW_FEATURES, make_features


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "loan_approval_dataset.csv"

MATCH = [
    "total_asset_value",
    "dti",
    "atr",
    "education",
    "self_employed",
    "no_of_dependents",
]

SCALED = [
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]


def match_supplied_split(x, y):
    """Match raw applications to the supplied train/test files by their features."""
    raw_keys = x[MATCH].copy()
    raw_keys[["dti", "atr"]] = raw_keys[["dti", "atr"]].round(8)
    raw_keys["raw_row"] = np.arange(len(x))

    if raw_keys.duplicated(MATCH).any():
        raise ValueError("Raw applicant keys are not unique; cannot match safely")

    matched = {}
    assigned = []

    for part in ("train", "test"):
        supplied = pd.read_csv(
            ROOT / "data" / f"{part}_engineered.csv"
        )

        keys = supplied[MATCH].copy()
        keys[["dti", "atr"]] = keys[["dti", "atr"]].round(8)

        if keys.duplicated(MATCH).any():
            raise ValueError(f"Duplicate {part} keys; cannot match safely")

        links = keys.merge(
            raw_keys,
            on=MATCH,
            how="left",
            validate="one_to_one",
            sort=False,
        )

        if links["raw_row"].isna().any() or len(links) != len(supplied):
            raise ValueError(
                f"Cannot match every supplied {part} row to the raw CSV"
            )

        positions = links["raw_row"].to_numpy(dtype=int)
        assigned.extend(positions)

        x_part = x.iloc[positions].reset_index(drop=True)
        y_part = y.iloc[positions].reset_index(drop=True)

        if not np.array_equal(
            y_part.to_numpy(),
            supplied["loan_status"].to_numpy(),
        ):
            raise ValueError(f"{part} labels disagree with raw CSV")

        for col in ("total_asset_value", "dti", "atr"):
            if not np.allclose(
                x_part[col].to_numpy(),
                supplied[col].to_numpy(),
                rtol=1e-7,
                atol=1e-6,
            ):
                raise ValueError(f"{part} {col} differs from supplied file")

        matched[part] = (x_part, y_part)

    if len(assigned) != len(x) or len(set(assigned)) != len(x):
        raise ValueError(
            "The supplied train/test files do not partition the raw CSV"
        )

    # Verify that the matched raw rows reproduce the supplied scaled columns.
    scaler = StandardScaler().fit(matched["train"][0][SCALED])

    for part in ("train", "test"):
        x_part, _ = matched[part]
        supplied_original = pd.read_csv(
            ROOT / "data" / f"{part}_original.csv"
        )

        transformed = scaler.transform(x_part[SCALED])

        if not np.allclose(
            transformed,
            supplied_original[SCALED].to_numpy(),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError(
                f"{part} standardized columns disagree with raw CSV"
            )

    return matched["train"], matched["test"]


def main():
    if not SOURCE.exists():
        raise SystemExit(
            "Missing data/loan_approval_dataset.csv"
        )

    raw = pd.read_csv(SOURCE)
    raw.columns = raw.columns.str.strip()

    for col in ("education", "self_employed", "loan_status"):
        raw[col] = raw[col].astype(str).str.strip()

    expected = set(RAW_FEATURES + ["loan_status", "loan_id"])

    if set(raw.columns) != expected or len(raw) != 4269:
        raise ValueError(
            "Expected the original 4,269-row, 13-column Kaggle file"
        )

    y = raw["loan_status"].map({
        "Approved": 1,
        "Rejected": 0,
    })

    if y.isna().any():
        raise ValueError("Unexpected target labels")

    x = make_features(raw)

    (x_train, y_train), (x_test, y_test) = match_supplied_split(x, y)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(x_train, y_train)

    report = {
        "note": "Supplied held-out split; no SMOTE or scaling; Approved=1",
        "train": score(model, x_train, y_train),
        "test": score(model, x_test, y_test),
    }

    (ROOT / "results").mkdir(exist_ok=True)

    (ROOT / "results" / "deployed_model_metrics.json").write_text(
        json.dumps(report, indent=2)
    )

    joblib.dump(
        {
            "model": model,
            "feature_order": FEATURES,
            "raw_columns": RAW_FEATURES,
        },
        ROOT / "loan_approval_model.joblib",
    )

    print(json.dumps(report, indent=2))
    print("Saved loan_approval_model.joblib")


if __name__ == "__main__":
    main()